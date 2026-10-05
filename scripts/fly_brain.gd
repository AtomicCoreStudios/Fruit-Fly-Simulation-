class_name FlyBrain
extends RefCounted
## Loads data/connectome.bin and runs the whole-brain LIF emulation on the GPU.
## Frame N submits a batch of 0.5 ms steps; frame N+1 syncs and reads back
## spike counts, so the GPU works while the scene renders.

const HEADER := 24

var meta: Dictionary
var n := 0
var e := 0
var n_pad := 0
var n_groups := 0
var n_regions := 0
var n_pops := 0
var pop_index := {}
var pop_region := PackedInt32Array()
var pop_size := PackedInt32Array()
var pop_hz := PackedFloat32Array()
var group_index := {}          # group name -> id
var group_size := PackedInt32Array()
var region_size := PackedInt32Array()
var multimesh_buffer := PackedFloat32Array()

var group_rate_in := PackedFloat32Array()   # Hz Poisson drive per group (write me)
var group_hz := PackedFloat32Array()        # measured output firing rate per group
var region_hz := PackedFloat32Array()
var activity_bytes := PackedByteArray()     # float per neuron, for the viewer
var total_spikes_per_s := 0.0
var sim_time_ms := 0.0
var time_scale := 1.0
var gpu_ms := 0.0

var _lif: Dictionary
var _rd: RenderingDevice
var _shader: RID
var _pipeline: RID
var _uset: RID
var _bufs := {}
var _step := 0
var _pending_steps := 0
var _step_accum := 0.0
var _submit_usec := 0
var _delay_steps := 1

# compound eye (per-facet vision -> per-photoreceptor rates), see shaders/eye.glsl
var eye_enabled := false
var eye_n_facets := 0
var eye_params := {"tau_adapt": 1.0, "r0": 40.0, "gain": 3.0, "r_max": 300.0}
var eye_facet_out := PackedFloat32Array()  # per facet: luminance, contrast, R1-6 rate, 0
var _eye_shader: RID
var _eye_pipeline: RID
var _eye_uset: RID
var _eye_pixels := PackedByteArray()
var _eye_dt := 0.0
var _eye_first := true
var _ext_cpu := PackedFloat32Array()      # [0,n): Poisson Hz, [n,2n): bias mV

# body-contact receptor neurons (taste etc.): CPU rates -> ext via shaders/sensor.glsl
const SENSOR_MAX := 8192
var _sn_shader: RID
var _sn_pipeline: RID
var _sn_uset: RID
var _sn_n := 0
var _sn_idx := PackedInt32Array()
var _sn_rate := PackedFloat32Array()

# graded optic lobe (FlyVis), see shaders/flyvis.glsl and tools/flyvis/
var flyvis_enabled := false
var flyvis_meta: Dictionary
var flyvis_gain := 60.0
var flyvis_rmax := 250.0
var _fv_shader: RID
var _fv_pipeline: RID
var _fv_uset: RID
var _fv_parity := 0
var _fv_every := 10          # LIF steps per FlyVis step (5 ms / 0.5 ms)

# graded optic lobe beyond FlyVis + graded VPN dendrites (shaders/graded.glsl, tools/graded/)
var graded_enabled := false
var graded_meta: Dictionary
var graded_c_mv := 10.0      # mV of soma bias per unit of VPN dendritic activity (free, swept)
var graded_gain := 60.0      # Hz per unit for non-VPN graded cells' spiking copies
var _gr_shader: RID
var _gr_pipeline: RID
var _gr_uset: RID


func load_connectome(bin_path: String, json_path: String) -> String:
	var jf := FileAccess.open(json_path, FileAccess.READ)
	if jf == null:
		return "missing %s - run: python tools/build_connectome.py synthetic" % json_path
	meta = JSON.parse_string(jf.get_as_text())
	_lif = meta["lif"]
	var f := FileAccess.open(bin_path, FileAccess.READ)
	if f == null:
		return "missing " + bin_path
	if f.get_buffer(4).get_string_from_ascii() != "FLYC":
		return "bad connectome magic"
	f.get_32()  # version
	n = f.get_32()
	e = f.get_32()
	n_pad = f.get_32()
	n_groups = f.get_32()
	n_regions = meta["regions"].size()
	var pop_b := f.get_buffer(n * 4)
	var group_b := f.get_buffer(n * 4)
	var rowptr_b := f.get_buffer((n + 1) * 4)
	var col_b := f.get_buffer(e * 4)
	var w_b := f.get_buffer(e * 4)
	multimesh_buffer = f.get_buffer(n * 64).to_float32_array()

	for i in meta["groups"].size():
		group_index[meta["groups"][i]["name"]] = i
	group_size.resize(n_groups)
	for i in n_groups:
		group_size[i] = int(meta["groups"][i]["count"])
	region_size.resize(n_regions)
	for i in n_regions:
		region_size[i] = int(meta["regions"][i]["count"])
	group_rate_in.resize(n_groups)
	group_hz.resize(n_groups)
	region_hz.resize(n_regions)

	var pops: Array = meta["populations"]
	n_pops = pops.size()
	pop_region.resize(n_pops)
	pop_size.resize(n_pops)
	pop_hz.resize(n_pops)
	for i in n_pops:
		pop_index[pops[i]["name"]] = i
		pop_region[i] = int(pops[i]["region"])
		pop_size[i] = int(pops[i]["count"])
	return _init_gpu(pop_b, group_b, rowptr_b, col_b, w_b)


func _init_gpu(pop_b, group_b, rowptr_b, col_b, w_b) -> String:
	_rd = RenderingServer.create_local_rendering_device()
	if _rd == null:
		return "No RenderingDevice - run with the Forward+ or Mobile renderer (not Compatibility/headless)."
	var src: RDShaderFile = load("res://shaders/lif.glsl")
	var spirv := src.get_spirv()
	if spirv.compile_error_compute != "":
		return "shader: " + spirv.compile_error_compute
	_shader = _rd.shader_create_from_spirv(spirv)
	_pipeline = _rd.compute_pipeline_create(_shader)

	var v0 := PackedFloat32Array()
	v0.resize(n)
	v0.fill(float(_lif["v_rest"]))
	var zeros_n := PackedByteArray()
	zeros_n.resize(n * 4)
	_delay_steps = maxi(1, int(round(float(_lif.get("delay_ms", _lif["dt_ms"])) / float(_lif["dt_ms"]))))
	var zeros_ring := PackedByteArray()
	zeros_ring.resize(n * 4 * (_delay_steps + 1))
	var zeros_pad := PackedByteArray()
	zeros_pad.resize(n_pad * 4)
	var zeros_g := PackedByteArray()
	zeros_g.resize(n_groups * 4)
	var zeros_r := PackedByteArray()
	zeros_r.resize(n_pops * 4)
	_ext_cpu.resize(2 * n)

	var specs := [
		["v", v0.to_byte_array()], ["g", zeros_n], ["refrac", zeros_n], ["g_in", zeros_ring],
		["spiked", zeros_n], ["group", group_b], ["pop", pop_b],
		["rates", group_rate_in.to_byte_array()], ["row_ptr", rowptr_b],
		["col", col_b if e > 0 else zeros_g], ["w", w_b if e > 0 else zeros_g],
		["activity", zeros_pad], ["gcount", zeros_g], ["rcount", zeros_r],
		["ext", _ext_cpu.to_byte_array()], ["v_kick", zeros_n], ["gap", _gap_bytes()],
		["phys", _phys_bytes()], ["st2", _zeros(n * 8)], ["homeo", _homeo_bytes()],
		["release", _release_init()], ["g_inh", _zeros(n * 4)], ["g_in_i", zeros_ring.duplicate()],
		["graded", _graded_bytes()],
	]
	var uniforms: Array[RDUniform] = []
	for b in specs.size():
		var data: PackedByteArray = specs[b][1]
		var buf := _rd.storage_buffer_create(data.size(), data)
		_bufs[specs[b][0]] = buf
		var u := RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = b
		u.add_id(buf)
		uniforms.append(u)
	_uset = _rd.uniform_set_create(uniforms, _shader, 0)
	activity_bytes.resize(n_pad * 4)
	return ""


var n_gap := 0


var physiology := true          # spontaneous activity + adaptation (data/physiology.bin)
var physiology_loaded := false
## Conductance-based synapses (default): reversal potentials. basis: E_exc ~0 mV for nicotinic ACh receptors
## (measured); E_inh -75 mV for Cl- channels (GABA-A/Rdl, GluCl, histamine-gated): approximate.
var conductance_synapses := true
var act_tau_ms := 60.0          # time constant of the per-neuron activity trace (readout/probe/homeostasis)
const E_EXC := 0.0
const E_INH := -75.0
var phys_ablate := []           # ablation switches: "noise", "adapt", "mu", "homeo"


func _zeros(nbytes: int) -> PackedByteArray:
	var z := PackedByteArray()
	z.resize(nbytes)
	return z


func _phys_bytes() -> PackedByteArray:
	# per neuron vec4(mu, sigma, b, tau_w); zeros (+tau 1) = original Shiu et al. behaviour
	var f := FileAccess.open("res://data/physiology.bin", FileAccess.READ)
	if physiology and f != null and f.get_length() == n * 16:
		physiology_loaded = true
		var pa := f.get_buffer(n * 16).to_float32_array()
		for i in n:
			if "mu" in phys_ablate: pa[4 * i] = 0.0
			if "noise" in phys_ablate: pa[4 * i + 1] = 0.0
			if "adapt" in phys_ablate: pa[4 * i + 2] = 0.0
		return pa.to_byte_array()
	var a := PackedFloat32Array()
	a.resize(n * 4)
	for i in n:
		a[4 * i + 3] = 1.0
	return a.to_byte_array()


var homeostasis_learning := false
var homeostasis_loaded := false


func _homeo_bytes() -> PackedByteArray:
	# vec2(target Hz, learned offset mV) per neuron; targets from build_physiology, offsets from a
	# previous --homeostasis warm-up (data/homeostasis_offsets.bin)
	var a := PackedFloat32Array()
	a.resize(n * 2)
	for i in n:
		a[2 * i] = -1.0
	var tf := FileAccess.open("res://data/physiology_targets.bin", FileAccess.READ)
	if physiology and tf != null and tf.get_length() == n * 4:
		var t := tf.get_buffer(n * 4).to_float32_array()
		for i in n:
			a[2 * i] = t[i]
	var of := FileAccess.open(homeo_path(), FileAccess.READ)
	if physiology and of != null and of.get_length() == n * 4 and not homeostasis_learning and not "homeo" in phys_ablate:
		var o := of.get_buffer(n * 4).to_float32_array()
		for i in n:
			a[2 * i + 1] = o[i]
		homeostasis_loaded = true
	return a.to_byte_array()


var graded_loaded := 0


func _graded_bytes() -> PackedByteArray:
	# per neuron vec4 (r_max Hz, gain Hz/mV, threshold mV, 0); r_max 0 = spiking (data/graded_release.bin,
	# tools/build_physiology.py)
	var f := FileAccess.open("res://data/graded_release.bin", FileAccess.READ)
	if physiology and f != null and f.get_length() == n * 16:
		var b := f.get_buffer(n * 16)
		var a := b.to_float32_array()
		for k in n:
			if a[4 * k] > 0.0:
				graded_loaded += 1
		return b
	return _zeros(n * 16)


var _release := PackedFloat32Array()
var _release_dirty := false


func _release_init() -> PackedByteArray:
	_release.resize(n)
	_release.fill(1.0)
	return _release.to_byte_array()


## Presynaptic release gain (neuromodulation): scales all outgoing synapses of the given neurons.
func set_release_gain(idx: PackedInt32Array, gain: float) -> void:
	if _release.size() != n:
		return
	for i in idx:
		if i >= 0 and i < n and _release[i] != gain:
			_release[i] = gain
			_release_dirty = true


## Homeostatic offsets are specific to the synapse model they were learned with.
func homeo_path() -> String:
	return "res://data/homeostasis_offsets%s.bin" % ("" if conductance_synapses else "_current")


## Save the learned homeostatic offsets (call at the end of a --homeostasis warm-up).
func save_homeostasis() -> void:
	if _pending_steps > 0:          # GPU work still in flight: finish it before reading buffers
		_rd.sync()
		_pending_steps = 0
	var h := _rd.buffer_get_data(_bufs["homeo"]).to_float32_array()
	var o := PackedFloat32Array()
	o.resize(n)
	for i in n:
		o[i] = h[2 * i + 1]
	var f := FileAccess.open(homeo_path(), FileAccess.WRITE)
	f.store_buffer(o.to_byte_array())
	f.close()


func _gap_bytes() -> PackedByteArray:
	# gap junctions from the connectome metadata: [pre, post, mV] -> ivec4(pre, post, micro-volts, 0)
	var gj: Array = meta.get("gap_junctions", [])
	n_gap = gj.size()
	var a := PackedInt32Array()
	for g in gj:
		a.append_array([int(g[0]), int(g[1]), int(round(float(g[2]) * 1000.0)), 0])
	if a.is_empty():
		a.append_array([0, 0, 0, 0])
	return a.to_byte_array()


func _push(mode: int) -> PackedByteArray:
	var pc := PackedByteArray()
	pc.resize(80)
	var dt := float(_lif["dt_ms"])
	pc.encode_u32(0, n)
	pc.encode_u32(4, mode)
	pc.encode_u32(8, _step)
	pc.encode_float(12, dt)
	pc.encode_float(16, float(_lif["tau_m_ms"]))
	pc.encode_float(20, exp(-dt / float(_lif["tau_syn_ms"])))
	pc.encode_float(24, float(_lif["v_rest"]))
	pc.encode_float(28, float(_lif["v_reset"]))
	pc.encode_float(32, float(_lif["v_th"]))
	pc.encode_float(36, float(_lif["refractory_ms"]))
	pc.encode_float(40, float(meta["w_poisson_mv"]))
	pc.encode_float(44, exp(-dt / act_tau_ms))
	pc.encode_u32(48, _delay_steps)
	pc.encode_u32(52, _delay_steps + 1)
	pc.encode_u32(56, 1 if _lif.get("reset_g_on_spike", false) else 0)
	pc.encode_u32(60, n_gap if mode == 2 else (1 if homeostasis_learning else 0))
	pc.encode_float(64, E_EXC)
	pc.encode_float(68, E_INH)
	pc.encode_u32(72, 1 if conductance_synapses else 0)
	return pc


## Call once per frame. Returns true when fresh readouts are available.
func tick(frame_dt: float) -> bool:
	var fresh := false
	if _pending_steps > 0:
		_rd.sync()
		gpu_ms = (Time.get_ticks_usec() - _submit_usec) / 1000.0
		_read_back(_pending_steps)
		_pending_steps = 0
		fresh = true

	var dt := float(_lif["dt_ms"])
	# cap the catch-up per frame: with a 50 ms cap one loading hitch locked the loop into
	# ~19 fps x 100 steps (still real time, but choppy); 34 ms lets it settle back to 60 fps
	_step_accum += minf(frame_dt, 0.034) * 1000.0 * time_scale / dt
	var steps := int(_step_accum)
	_step_accum -= steps
	steps = mini(steps, 200)
	if steps <= 0:
		return fresh

	_rd.buffer_update(_bufs["rates"], 0, n_groups * 4, group_rate_in.to_byte_array())
	var groups_x := int(ceil(n / 256.0))
	if _sn_n > 0:
		_rd.buffer_update(_bufs["sn_idx"], 0, _sn_n * 4, _sn_idx.to_byte_array())
		_rd.buffer_update(_bufs["sn_rate"], 0, _sn_n * 4, _sn_rate.to_byte_array())
	var eye_now := eye_enabled and _eye_pixels.size() > 0
	if eye_now:
		_rd.buffer_update(_bufs["eye_pix"], 0, _eye_pixels.size(), _eye_pixels)
	if _release_dirty:
		_rd.buffer_update(_bufs["release"], 0, n * 4, _release.to_byte_array())
		_release_dirty = false
	var cl := _rd.compute_list_begin()
	if _sn_n > 0:
		var spc := PackedByteArray()
		spc.resize(16)
		spc.encode_u32(0, _sn_n)
		_rd.compute_list_bind_compute_pipeline(cl, _sn_pipeline)
		_rd.compute_list_bind_uniform_set(cl, _sn_uset, 0)
		_rd.compute_list_set_push_constant(cl, spc, spc.size())
		_rd.compute_list_dispatch(cl, int(ceil(_sn_n / 64.0)), 1, 1)
		_rd.compute_list_add_barrier(cl)
	if eye_now:
		var epc := PackedByteArray()
		epc.resize(32)
		epc.encode_u32(0, eye_n_facets)
		epc.encode_u32(4, 1 if _eye_first else 0)
		epc.encode_float(8, maxf(_eye_dt, 1e-4))
		epc.encode_float(12, eye_params["tau_adapt"])
		epc.encode_float(16, eye_params["r0"])
		epc.encode_float(20, eye_params["gain"])
		epc.encode_float(24, eye_params["r_max"])
		_rd.compute_list_bind_compute_pipeline(cl, _eye_pipeline)
		_rd.compute_list_bind_uniform_set(cl, _eye_uset, 0)
		_rd.compute_list_set_push_constant(cl, epc, epc.size())
		_rd.compute_list_dispatch(cl, int(ceil(eye_n_facets / 64.0)), 1, 1)
		_rd.compute_list_add_barrier(cl)
		_eye_first = false
		_eye_pixels = PackedByteArray()
	_rd.compute_list_bind_compute_pipeline(cl, _pipeline)
	_rd.compute_list_bind_uniform_set(cl, _uset, 0)
	for s in steps:
		if flyvis_enabled and _step % _fv_every == 0:
			_dispatch_flyvis(cl)
		for mode in (3 if n_gap > 0 else 2):
			var pc := _push(mode)
			_rd.compute_list_set_push_constant(cl, pc, pc.size())
			_rd.compute_list_dispatch(cl, groups_x if mode < 2 else int(ceil(n_gap / 256.0)), 1, 1)
			_rd.compute_list_add_barrier(cl)
		_step += 1
	_rd.compute_list_end()
	_rd.submit()
	_submit_usec = Time.get_ticks_usec()
	_pending_steps = steps
	return fresh


func _read_back(steps: int) -> void:
	var secs := steps * float(_lif["dt_ms"]) / 1000.0
	sim_time_ms += secs * 1000.0
	var gc := _rd.buffer_get_data(_bufs["gcount"]).to_int32_array()
	var rc := _rd.buffer_get_data(_bufs["rcount"]).to_int32_array()
	_rd.buffer_clear(_bufs["gcount"], 0, n_groups * 4)
	_rd.buffer_clear(_bufs["rcount"], 0, n_pops * 4)
	var total := 0
	for i in n_groups:
		group_hz[i] = gc[i] / (maxf(group_size[i], 1) * secs)
	region_hz.fill(0.0)
	for i in n_pops:
		pop_hz[i] = rc[i] / (maxf(pop_size[i], 1) * secs)
		region_hz[pop_region[i]] += rc[i]
		total += rc[i]
	for i in n_regions:
		region_hz[i] /= maxf(region_size[i], 1) * secs
	total_spikes_per_s = total / secs
	activity_bytes = _rd.buffer_get_data(_bufs["activity"])
	if eye_enabled:
		eye_facet_out = _rd.buffer_get_data(_bufs["eye_out"]).to_float32_array()


## Set up the compound eye from data/eye_drive.json (tools/build_eye_drive.py).
func init_eye(d: Dictionary) -> String:
	var src: RDShaderFile = load("res://shaders/eye.glsl")
	var spirv := src.get_spirv()
	if spirv.compile_error_compute != "":
		return "eye shader: " + spirv.compile_error_compute
	_eye_shader = _rd.shader_create_from_spirv(spirv)
	_eye_pipeline = _rd.compute_pipeline_create(_eye_shader)
	eye_n_facets = int(d["n_facets"])
	var px: int = int(d["cube_px"])
	var pix := PackedByteArray()
	pix.resize(6 * px * px * 4)
	var adapt := PackedByteArray()
	adapt.resize(eye_n_facets * 16)
	var specs := [
		["eye_pix", pix], ["eye_kptr", PackedInt32Array(d["k_ptr"]).to_byte_array()],
		["eye_kpix", PackedInt32Array(d["k_pix"]).to_byte_array()],
		["eye_kw", PackedFloat32Array(d["k_w"]).to_byte_array()], ["eye_adapt", adapt],
		["eye_pptr", PackedInt32Array(d["pr_ptr"]).to_byte_array()],
		["eye_pneu", PackedInt32Array(d["pr_neuron"]).to_byte_array()],
		["eye_pchan", PackedInt32Array(d["pr_chan"]).to_byte_array()],
		["ext", null], ["eye_out", adapt.duplicate()],
	]
	var uniforms: Array[RDUniform] = []
	for b in specs.size():
		var nm: String = specs[b][0]
		if not _bufs.has(nm):
			var data: PackedByteArray = specs[b][1]
			_bufs[nm] = _rd.storage_buffer_create(data.size(), data)
		var u := RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = b
		u.add_id(_bufs[nm])
		uniforms.append(u)
	_eye_uset = _rd.uniform_set_create(uniforms, _eye_shader, 0)
	eye_enabled = true
	return ""


func _dispatch_flyvis(cl: int) -> void:
	var nn: int = int(flyvis_meta["n_nodes"])
	var n_out: int = int(flyvis_meta["n_out"])
	_rd.compute_list_bind_compute_pipeline(cl, _fv_pipeline)
	_rd.compute_list_bind_uniform_set(cl, _fv_uset, 0)
	for mode in 2:
		var pc := PackedByteArray()
		pc.resize(32)
		pc.encode_u32(0, nn)
		pc.encode_u32(4, mode)
		# mode 0 reads half `parity`, writes the other; mode 1 reads the freshly written half
		pc.encode_u32(8, _fv_parity if mode == 0 else 1 - _fv_parity)
		pc.encode_u32(12, n_out)
		pc.encode_float(16, float(flyvis_meta["dt_s"]))
		pc.encode_float(20, flyvis_gain)
		pc.encode_float(24, flyvis_rmax)
		_rd.compute_list_set_push_constant(cl, pc, pc.size())
		_rd.compute_list_dispatch(cl, int(ceil((2 * nn if mode == 0 else n_out) / 256.0)), 1, 1)
		_rd.compute_list_add_barrier(cl)
	_fv_parity = 1 - _fv_parity
	if graded_enabled:
		_dispatch_graded(cl, _fv_parity * 2 * nn)
	_rd.compute_list_bind_compute_pipeline(cl, _pipeline)
	_rd.compute_list_bind_uniform_set(cl, _uset, 0)


func _dispatch_graded(cl: int, fv_off: int) -> void:
	var nd: int = int(graded_meta["n_d"])
	var ng: int = int(graded_meta["n_g"])
	_rd.compute_list_bind_compute_pipeline(cl, _gr_pipeline)
	_rd.compute_list_bind_uniform_set(cl, _gr_uset, 0)
	for mode in 3:
		var pc := PackedByteArray()
		pc.resize(32)
		pc.encode_u32(0, nd)
		pc.encode_u32(4, ng)
		pc.encode_u32(8, mode)
		pc.encode_u32(12, fv_off)
		pc.encode_float(16, float(graded_meta["dt_s"]))
		pc.encode_float(20, graded_gain)
		pc.encode_float(24, graded_c_mv)
		pc.encode_u32(28, n)
		_rd.compute_list_set_push_constant(cl, pc, pc.size())
		# mode 1 runs 32 threads per unit: 8 units per 256-thread workgroup
		var groups := int(ceil(nd / 256.0)) if mode == 0 else (int(ceil(ng / 8.0)) if mode == 1 else int(ceil(ng / 256.0)))
		_rd.compute_list_dispatch(cl, groups, 1, 1)
		_rd.compute_list_add_barrier(cl)


## Graded optic lobe beyond FlyVis + VPN dendrites (data/graded_gpu.*); needs init_flyvis() first.
func init_graded() -> String:
	if not flyvis_enabled:
		return "graded model needs FlyVis"
	var jf := FileAccess.open("res://data/graded_gpu.json", FileAccess.READ)
	var bf := FileAccess.open("res://data/graded_gpu.bin", FileAccess.READ)
	var wf := FileAccess.open("res://data/lif_w_graded.bin", FileAccess.READ)
	if jf == null or bf == null or wf == null:
		return "missing data/graded_gpu.* or lif_w_graded.bin (run tools/graded/export_godot.py)"
	graded_meta = JSON.parse_string(jf.get_as_text())
	var raw := bf.get_buffer(bf.get_length())
	var blob := func(name: String) -> PackedByteArray:
		var o: Array = graded_meta["offsets"][name]
		return raw.slice(int(o[0]), int(o[0]) + int(o[1]))
	var src: RDShaderFile = load("res://shaders/graded.glsl")
	if src == null:
		return "shaders/graded.glsl not imported (run Godot --headless --import)"
	var spirv := src.get_spirv()
	if spirv.compile_error_compute != "":
		return "graded shader: " + spirv.compile_error_compute
	_gr_shader = _rd.shader_create_from_spirv(spirv)
	_gr_pipeline = _rd.compute_pipeline_create(_gr_shader)
	var nd: int = int(graded_meta["n_d"])
	var ng: int = int(graded_meta["n_g"])
	var zA := PackedByteArray()
	zA.resize((nd + ng) * 4)
	var zV := PackedByteArray()
	zV.resize(ng * 4)
	var specs := [["gr_A", zA], ["gr_V", zV], ["gr_tau", blob.call("tau")], ["gr_bias", blob.call("bias")],
		["gr_row", blob.call("row_ptr")], ["gr_col", blob.call("col")], ["gr_w", blob.call("w")],
		["gr_dnode", blob.call("d_node")], ["fv_state", null], ["gr_fw", blob.call("g_fw")],
		["gr_vpn", blob.call("g_vpn")], ["ext", null]]
	var uniforms: Array[RDUniform] = []
	for b in specs.size():
		var nm: String = specs[b][0]
		if not _bufs.has(nm):
			var data: PackedByteArray = specs[b][1]
			_bufs[nm] = _rd.storage_buffer_create(data.size(), data)
		var u := RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = b
		u.add_id(_bufs[nm])
		uniforms.append(u)
	_gr_uset = _rd.uniform_set_create(uniforms, _gr_shader, 0)
	# silence LIF synapses now carried by the graded model (see tools/graded/export_godot.py)
	var wb := wf.get_buffer(wf.get_length())
	if wb.size() != e * 4:
		return "lif_w_graded.bin does not match the loaded connectome"
	_rd.buffer_update(_bufs["w"], 0, wb.size(), wb)
	graded_enabled = true
	return ""


## Load the FlyVis graded optic lobe (data/flyvis_gpu.*); needs init_eye() first.
func init_flyvis() -> String:
	if not eye_enabled:
		return "FlyVis needs the compound eye"
	var jf := FileAccess.open("res://data/flyvis_gpu.json", FileAccess.READ)
	var bf := FileAccess.open("res://data/flyvis_gpu.bin", FileAccess.READ)
	if jf == null or bf == null:
		return "missing data/flyvis_gpu.* (run tools/flyvis/export_godot.py)"
	flyvis_meta = JSON.parse_string(jf.get_as_text())
	var raw := bf.get_buffer(bf.get_length())
	var blob := func(name: String) -> PackedByteArray:
		var o: Array = flyvis_meta["offsets"][name]
		return raw.slice(int(o[0]), int(o[0]) + int(o[1]))
	var src: RDShaderFile = load("res://shaders/flyvis.glsl")
	if src == null:
		return "shaders/flyvis.glsl not imported (run Godot --headless --import)"
	var spirv := src.get_spirv()
	if spirv.compile_error_compute != "":
		return "flyvis shader: " + spirv.compile_error_compute
	_fv_shader = _rd.shader_create_from_spirv(spirv)
	_fv_pipeline = _rd.compute_pipeline_create(_fv_shader)
	# state: 2 ping-pong halves x 2 eyes, starting at the grey-adapted steady state
	var vg: PackedByteArray = blob.call("v_grey")
	var state := PackedByteArray()
	for k in 4:
		state.append_array(vg)
	var specs := [["fv_state", state], ["fv_bias", blob.call("bias")], ["fv_tau", blob.call("tau")],
		["fv_cell", blob.call("cell")], ["fv_fptr", blob.call("filt_ptr")], ["fv_filt", blob.call("filt")],
		["fv_in", blob.call("in_facet")], ["eye_out", null], ["fv_pairs", blob.call("out_pairs")],
		["fv_base", blob.call("out_base")], ["ext", null], ["fv_grid", blob.call("grid")]]
	var uniforms: Array[RDUniform] = []
	for b in specs.size():
		var nm: String = specs[b][0]
		if not _bufs.has(nm):
			var data: PackedByteArray = specs[b][1]
			_bufs[nm] = _rd.storage_buffer_create(data.size(), data)
		var u := RDUniform.new()
		u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
		u.binding = b
		u.add_id(_bufs[nm])
		uniforms.append(u)
	_fv_uset = _rd.uniform_set_create(uniforms, _fv_shader, 0)
	flyvis_enabled = true
	return ""


## Receptor neurons driven by the body (taste contact etc.): model indices + Poisson rates (Hz).
## Send the full list every frame; neurons left out keep their last rate.
func set_sensor_rates(idx: PackedInt32Array, rates: PackedFloat32Array) -> void:
	if not _sn_shader.is_valid():
		var src: RDShaderFile = load("res://shaders/sensor.glsl")
		if src == null:
			push_error("shaders/sensor.glsl not imported")
			return
		_sn_shader = _rd.shader_create_from_spirv(src.get_spirv())
		_sn_pipeline = _rd.compute_pipeline_create(_sn_shader)
		var z := PackedByteArray()
		z.resize(SENSOR_MAX * 4)
		_bufs["sn_idx"] = _rd.storage_buffer_create(z.size(), z)
		_bufs["sn_rate"] = _rd.storage_buffer_create(z.size(), z)
		var uniforms: Array[RDUniform] = []
		for b in 3:
			var u := RDUniform.new()
			u.uniform_type = RenderingDevice.UNIFORM_TYPE_STORAGE_BUFFER
			u.binding = b
			u.add_id(_bufs[["sn_idx", "sn_rate", "ext"][b]])
			uniforms.append(u)
		_sn_uset = _rd.uniform_set_create(uniforms, _sn_shader, 0)
	_sn_n = mini(idx.size(), SENSOR_MAX)
	_sn_idx = idx.slice(0, _sn_n)
	_sn_rate = rates.slice(0, _sn_n)


## Hand over the latest 6-face cube render (RGBA8, face-major, rows top to bottom).
func set_eye_pixels(bytes: PackedByteArray, dt: float) -> void:
	_eye_pixels = bytes
	_eye_dt = dt


## Resting depolarisation (mV above rest) for graded cells, e.g. lamina neurons.
func set_neuron_bias(indices: Array, mv: float) -> void:
	for i in indices:
		_ext_cpu[n + int(i)] = mv
	_rd.buffer_update(_bufs["ext"], n * 4, n * 4, _ext_cpu.slice(n).to_byte_array())


func gid(name: String) -> int:
	return group_index.get(name, 0)


func set_input(name: String, hz: float) -> void:
	var i: int = group_index.get(name, 0)
	if i > 0:
		group_rate_in[i] = maxf(hz, 0.0)


func rate(name: String) -> float:
	var i: int = group_index.get(name, 0)
	return group_hz[i] if i > 0 else 0.0


func pop_rate(name: String) -> float:
	return pop_hz[pop_index[name]] if pop_index.has(name) else -1.0


func free_gpu() -> void:
	if _rd == null:
		return
	if _pending_steps > 0:
		_rd.sync()
	if _eye_shader.is_valid():
		_rd.free_rid(_eye_shader)
	if _fv_shader.is_valid():
		_rd.free_rid(_fv_shader)
	if _gr_shader.is_valid():
		_rd.free_rid(_gr_shader)
	if _sn_shader.is_valid():
		_rd.free_rid(_sn_shader)
	for b in _bufs.values():
		_rd.free_rid(b)
	_rd.free_rid(_shader)
	_rd.free()
	_rd = null
