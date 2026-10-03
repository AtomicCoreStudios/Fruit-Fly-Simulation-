class_name Fly
extends Node3D
## The embodied fly: loads the NeuroMechFly micro-CT body (assets/fly/fly.glb, built in
## Blender by tools/blender/), turns the world into sensory spike rates for the brain, and
## turns descending-neuron firing into movement. Falls back to a placeholder body if the
## glTF is missing.

const SECTORS := 6
const COMPASS := 8
const SECTOR_W := PI / SECTORS          # each eye sees 180 deg split into 6 sectors
const MAX_SPEED := 2.0                  # arena units (cm) per second
const MAX_TURN := 4.0                   # rad/s

var brain: FlyBrain
var world               # Main node (odour/wind/landmark queries)

var hunger := 0.8
var speed := 0.0
var turn_rate := 0.0
var airborne := 0.0
var proboscis := 0.0
var feeding := false
var escapes := 0
var opto_p9 := false          # optogenetic activation of P9 / DNp09 (Bidaye et al. 2020)
var _escape_cooldown := 0.0

var sense := {}         # last sensory values, for the HUD
var motor := {}         # smoothed motor readouts
var _lum_prev := {}
var _odour_adapt := {}
var _phase := 0.0
var _legs: Array[Node3D] = []
var _wings: Array[Node3D] = []
var _prob: Node3D
var _escape_dir := Vector3.ZERO

var vision_mode := "facets"   # "facets": 1,709 measured ommatidia; "sectors": old 2x6 sectors
var lamina_bias_mv := 9.0
var columnar_bias_mv := 0.0
var optic_lobe := "flyvis"    # "flyvis": graded FlyVis optic lobe drives FlyWire columnar cells; "lif": spiking only
var flyvis_gain := 60.0
var graded_vpn := true         # graded optic lobe beyond FlyVis + two-compartment VPNs
var graded_c_mv := 10.0
var tethered := false         # body held in place (like a tethered-fly rig); brain and senses run
var eye: FlyEye
var seg := {}                 # body segment name -> Node3D (FlyGym names, e.g. "lf_tibia")
var _rest := {}               # segment -> rest Basis, for animation on top of the pose
var _body: Node3D


func _ready() -> void:
	if ResourceLoader.exists("res://assets/fly/fly.glb"):
		_load_body()
	else:
		_build_body()
	if vision_mode == "facets" and seg.has("c_head") and brain.meta.get("source", "") == "flywire_v783":
		eye = FlyEye.new()
		add_child(eye)
		var use_fv := optic_lobe == "flyvis" and FileAccess.file_exists("res://data/flyvis_gpu.bin")
		# with FlyVis the lamina is driven by the graded model, so no LIF resting bias there
		var err := eye.setup(brain, seg["c_head"], 0.0 if use_fv else lamina_bias_mv, columnar_bias_mv)
		if err == "" and use_fv:
			brain.flyvis_gain = flyvis_gain
			err = brain.init_flyvis()
			if err == "" and graded_vpn and FileAccess.file_exists("res://data/graded_gpu.bin"):
				brain.graded_c_mv = graded_c_mv
				brain.graded_gain = flyvis_gain
				err = brain.init_graded()
		if err != "":
			push_warning("compound eye disabled: " + err)
			eye.queue_free()
			eye = null
	for k in ["fwd_L", "fwd_R", "turn_L", "turn_R", "back", "gf", "mn9"]:
		motor[k] = 0.0


# ---------------------------------------------------------------- body
func _mat(c: Color, rough := 0.6, metal := 0.0) -> StandardMaterial3D:
	var m := StandardMaterial3D.new()
	m.albedo_color = c
	m.roughness = rough
	m.metallic = metal
	return m


func _blob(parent: Node3D, pos: Vector3, size: Vector3, m: Material) -> MeshInstance3D:
	var mi := MeshInstance3D.new()
	var s := SphereMesh.new()
	s.radius = 0.5
	s.height = 1.0
	s.radial_segments = 16
	s.rings = 8
	mi.mesh = s
	mi.material_override = m
	mi.position = pos
	mi.scale = size
	parent.add_child(mi)
	return mi


func _load_body() -> void:
	# glTF units are mm (Blender), arena units are cm; FlyGym +X anterior -> Godot -Z forward.
	_body = (load("res://assets/fly/fly.glb") as PackedScene).instantiate()
	_body.scale = Vector3.ONE * 0.1
	_body.rotation.y = PI / 2
	_body.position.y = -0.06
	add_child(_body)
	var stack: Array[Node] = [_body]
	while stack.size() > 0:
		var nd: Node = stack.pop_back()
		stack.append_array(nd.get_children())
		if nd is MeshInstance3D:
			(nd as MeshInstance3D).layers = FlyEye.BODY_LAYER
			# The fly's ~0.3 mm self-shadow is below shadow-map resolution and pops in and out
			# when the shadow cascade refits (seen by the ventral facets as a floor flicker).
			# Not rendered. basis: approximate
			(nd as MeshInstance3D).cast_shadow = GeometryInstance3D.SHADOW_CASTING_SETTING_OFF
			if str(nd.name).ends_with("wing_mesh"):
				var wm := StandardMaterial3D.new()   # glTF import drops the Blender wing alpha
				wm.albedo_color = Color(0.8, 0.8, 0.9, 0.3)
				wm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
				wm.cull_mode = BaseMaterial3D.CULL_DISABLED
				wm.roughness = 0.2
				(nd as MeshInstance3D).material_override = wm
		elif nd is Node3D and not nd.name.ends_with("_mesh"):
			seg[str(nd.name)] = nd
			_rest[str(nd.name)] = (nd as Node3D).basis
	position.y = 0.06


func _animate_body(t_air: float) -> void:
	# Cosmetic only: leg swing and wing beat are not driven by a motor model (no nerve cord yet).
	if seg.is_empty():
		return
	for leg in ["lf", "lm", "lh", "rf", "rm", "rh"]:
		var node: Node3D = seg.get(leg + "_coxa")
		if node == null:
			continue
		var tri := 0.0 if leg in ["lf", "rm", "lh"] else PI
		node.basis = _rest[leg + "_coxa"] * Basis(Vector3(0, 1, 0), sin(_phase + tri) * 0.3 * minf(absf(speed) + absf(turn_rate) * 0.2, 1.0))
	for w in ["l_wing", "r_wing"]:
		if seg.has(w):
			var flap := sin(Time.get_ticks_msec() * 0.9) * 1.0 if t_air > 0.0 else 0.0
			seg[w].basis = _rest[w] * Basis(Vector3(1, 0, 0), flap * (1 if w == "l_wing" else -1))
	if seg.has("c_rostrum"):
		seg["c_rostrum"].basis = _rest["c_rostrum"] * Basis(Vector3(0, 1, 0), proboscis * 0.9)


func _build_body() -> void:
	var body := _mat(Color(0.42, 0.33, 0.22))
	var dark := _mat(Color(0.12, 0.1, 0.08))
	_blob(self, Vector3(0, 0.0, 0.0), Vector3(0.11, 0.09, 0.13), body)          # thorax
	_blob(self, Vector3(0, -0.005, 0.15), Vector3(0.09, 0.075, 0.17), dark)      # abdomen
	_blob(self, Vector3(0, 0.005, -0.1), Vector3(0.08, 0.07, 0.06), body)        # head
	var eye := _mat(Color(0.75, 0.08, 0.05), 0.25)
	for sx in [-1, 1]:
		_blob(self, Vector3(sx * 0.037, 0.01, -0.105), Vector3(0.04, 0.055, 0.05), eye)
		var ant := _blob(self, Vector3(sx * 0.012, 0.02, -0.135), Vector3(0.01, 0.03, 0.01), dark)
		ant.rotation.x = -0.6
		# wings
		var wp := Node3D.new()
		wp.position = Vector3(sx * 0.03, 0.04, -0.01)
		add_child(wp)
		var wing := MeshInstance3D.new()
		var q := QuadMesh.new()
		q.size = Vector2(0.09, 0.24)
		wing.mesh = q
		var wm := _mat(Color(0.85, 0.9, 1.0, 0.35), 0.1)
		wm.transparency = BaseMaterial3D.TRANSPARENCY_ALPHA
		wm.cull_mode = BaseMaterial3D.CULL_DISABLED
		wing.material_override = wm
		wing.rotation.x = -PI / 2
		wing.position = Vector3(sx * 0.03, 0, 0.11)
		wp.add_child(wing)
		wp.rotation.y = sx * 0.18
		_wings.append(wp)
		# legs: front, mid, hind
		for j in 3:
			var pivot := Node3D.new()
			pivot.position = Vector3(sx * 0.035, -0.02, -0.035 + j * 0.035)
			add_child(pivot)
			var seg := MeshInstance3D.new()
			var bm := BoxMesh.new()
			bm.size = Vector3(0.13, 0.008, 0.008)
			seg.mesh = bm
			seg.material_override = dark
			seg.position = Vector3(sx * 0.055, -0.02, 0)
			seg.rotation.z = sx * -0.45
			pivot.add_child(seg)
			pivot.set_meta("sx", sx)
			pivot.set_meta("base", (j - 1) * 0.55 * sx)
			# tripod gait: L1,R2,L3 vs R1,L2,R3
			pivot.set_meta("tripod", (j + (0 if sx > 0 else 1)) % 2)
			_legs.append(pivot)
	_prob = Node3D.new()
	_prob.position = Vector3(0, -0.03, -0.12)
	add_child(_prob)
	var pm := _blob(_prob, Vector3(0, -0.03, 0), Vector3(0.012, 0.06, 0.012), dark)
	pm.name = "proboscis"
	position.y = 0.06


# ---------------------------------------------------------------- sensing
func _azimuth_to(p: Vector3) -> float:
	var l := to_local(p)
	return atan2(-l.x, -l.z)  # 0 = straight ahead, + = left


static func _wrap(a: float) -> float:
	return wrapf(a, -PI, PI)


func _sector_luminance(eye_sign: float, k: int) -> float:
	# sector centre azimuth; left eye covers 0..+180, right 0..-180
	var centre := eye_sign * (k + 0.5) * SECTOR_W
	var sun_rel := _wrap(world.sun_azimuth - rotation.y)
	var d := _wrap(sun_rel - centre)
	var lum := 0.75 + 0.5 * exp(-d * d / (2.0 * 0.9 * 0.9))
	for ob in world.visual_objects():
		var p: Vector3 = ob["pos"]
		var r: float = ob["radius"]
		var dist := maxf(Vector2(p.x - position.x, p.z - position.z).length(), r + 0.01)
		var width := 2.0 * atan(r / dist)
		var az := _wrap(_azimuth_to(p) - centre)
		# overlap of the object's angular extent with this sector
		var lo := maxf(az - width / 2, -SECTOR_W / 2)
		var hi := minf(az + width / 2, SECTOR_W / 2)
		if hi <= lo:
			continue
		var elev := clampf(atan2(ob["height"] - position.y, dist) / 0.8, 0.0, 1.0)
		lum -= (hi - lo) / SECTOR_W * elev * ob["darkness"]
	return clampf(lum, 0.0, 1.5)


func update_senses(dt: float) -> void:
	var b := brain
	var head := global_transform * Vector3(0, 0, -0.12)
	var left := -global_transform.basis.x.normalized()
	var fwd := -global_transform.basis.z.normalized()
	var loom := 0.0
	var odour_sum := {"food": 0.0, "aversive": 0.0}
	# photoreceptors adapt to the mean light level, so they signal local contrast
	var lums := {}
	var mean := 0.0
	for side in ["L", "R"]:
		for k in SECTORS:
			var l := _sector_luminance(1.0 if side == "L" else -1.0, k)
			lums["%s%d" % [side, k]] = l
			mean += l / (2.0 * SECTORS)
	for side in ["L", "R"]:
		var sgn := 1.0 if side == "L" else -1.0
		for k in SECTORS:
			var key := "%s%d" % [side, k]
			var lum: float = lums[key]
			var prev: float = _lum_prev.get(key, lum)
			# OFF-transient (darkening) drives the loom pathway; an efference copy of
			# the fly's own turning suppresses it, as in real saccade suppression
			var trans := maxf(prev - lum, 0.0) / maxf(dt, 1e-3) / (1.0 + absf(turn_rate) * 0.5)
			_lum_prev[key] = lum
			if eye != null:
				# facet vision drives the real photoreceptors; the old sector groups stay silent
				b.set_input("eye_%s_tonic_%d" % [side, k], 0.0)
				b.set_input("eye_%s_transient_%d" % [side, k], 0.0)
				b.set_input("eye_%s_on_%d" % [side, k], 0.0)
			else:
				b.set_input("eye_%s_tonic_%d" % [side, k], clampf(90.0 + 300.0 * (lum - mean), 5.0, 300.0))
				b.set_input("eye_%s_transient_%d" % [side, k], clampf((trans - 0.3) * 300.0, 0.0, 300.0))
			sense["lum_" + key] = lum
			loom += trans
		# antennae sample the odour plume a little ahead and to each side
		var ant := head + left * sgn * 0.3 + fwd * 0.15
		var food: float = world.odour_at(ant, "food")
		var bad: float = world.odour_at(ant, "aversive")
		if not _odour_adapt.has("food"):
			_odour_adapt = {"food": maxf(food, 0.03), "aversive": maxf(bad, 0.03)}
		# ORNs: steep Hill dose-response around a slowly adapting set point, so
		# they report left/right contrast instead of saturating near the source
		b.set_input("orn_%s_attractive" % side, 5.0 + 250.0 * _hill(food, _odour_adapt["food"]))
		b.set_input("orn_%s_aversive" % side, 5.0 + 250.0 * _hill(bad, _odour_adapt["aversive"]))
		odour_sum["food"] += food * 0.5
		odour_sum["aversive"] += bad * 0.5
		sense["odour_" + side] = food
		sense["bad_" + side] = bad
		# Johnston's organ: wind blowing onto this side of the head
		var wind: Vector3 = world.wind
		var from_side := maxf(0.0, (-wind.normalized()).dot(left * sgn)) if wind.length() > 0.01 else 0.0
		var jo := 10.0 + wind.length() * (60.0 + 220.0 * from_side)
		b.set_input("jo_" + side, jo)
		sense["jo_" + side] = jo
		# taste: tarsal / labellar GRNs touching a patch
		var patch: String = world.patch_under(global_position) if airborne <= 0.0 else ""
		b.set_input("grn_%s_sweet" % side, 220.0 if patch == "food" else 0.0)
		b.set_input("grn_%s_bitter" % side, 220.0 if patch == "aversive" else 0.0)
		sense["taste"] = patch
	sense["loom"] = loom
	if eye != null:
		eye.update(dt)
	for k in odour_sum:
		if not _odour_adapt.has(k):
			_odour_adapt[k] = maxf(odour_sum[k], 0.03)
		_odour_adapt[k] = lerpf(_odour_adapt[k], maxf(odour_sum[k], 0.03), 1.0 - exp(-dt / 2.0))
	# compass: sun position relative to heading -> central-complex wedges
	var sun_rel := _wrap(world.sun_azimuth - rotation.y)
	for k in COMPASS:
		var d := _wrap(sun_rel - TAU * k / COMPASS)
		b.set_input("compass_%d" % k, 180.0 * exp(-d * d / (2.0 * 0.5 * 0.5)))
	# internal state: background tone and hunger-driven exploration
	b.set_input("bg_off", 110.0)
	b.set_input("bg_spont", 30.0)
	b.set_input("drive_explore", 40.0 + 90.0 * hunger)
	# optogenetics: Poisson drive straight into the forward-walking descending neurons
	b.set_input("dn_forward_L", 80.0 if opto_p9 else 0.0)
	b.set_input("dn_forward_R", 80.0 if opto_p9 else 0.0)


static func _hill(c: float, k: float) -> float:
	var x := pow(c / k, 3.0)
	return x / (1.0 + x)


# ---------------------------------------------------------------- acting
func update_motor(dt: float) -> void:
	var b := brain
	var a := 1.0 - exp(-dt / 0.25)
	var raw := {
		"fwd_L": b.rate("dn_forward_L"), "fwd_R": b.rate("dn_forward_R"),
		"turn_L": b.rate("dn_turn_L"), "turn_R": b.rate("dn_turn_R"),
		"back": b.rate("dn_backward"), "mn9": b.rate("mn9_proboscis"),
	}
	for k in raw:
		motor[k] = lerpf(motor[k], raw[k], a)
	motor["gf"] = b.rate("dn_giant_fiber")

	# Giant fibre spike -> escape take-off, away from whatever loomed
	_escape_cooldown -= dt
	if tethered:
		if motor["gf"] > 0.0 and _escape_cooldown <= 0.0:
			escapes += 1          # giant-fibre spike = escape command, counted but body held
			_escape_cooldown = 3.0
			print("ESCAPE (giant fibre) at t=%.2fs" % (brain.sim_time_ms / 1000.0))
		speed = 0.0
		turn_rate = 0.0
		return
	if motor["gf"] > 0.0 and airborne <= 0.0 and _escape_cooldown <= 0.0:
		airborne = 1.4
		_escape_cooldown = 3.0
		escapes += 1
		var threat: Vector3 = world.threat_position()
		var away := global_position - threat
		away.y = 0
		_escape_dir = away.normalized() if away.length() > 0.01 else global_transform.basis.z
	if airborne > 0.0:
		airborne -= dt
		var t := 1.4 - airborne
		global_position += _escape_dir * 4.0 * dt
		position.y = 0.06 + sin(clampf(t / 1.4, 0, 1) * PI) * 1.2
		look_at(global_position + _escape_dir, Vector3.UP)
		_animate_body(airborne)
		for wgt in _wings:
			wgt.rotation.z = sin(Time.get_ticks_msec() * 0.9) * 1.0
		speed = 0.0
		_clamp_to_arena()
		return
	for wgt in _wings:
		wgt.rotation.z = 0.0
	position.y = 0.06

	# descending-neuron rate -> body command; calibration comes from the connectome metadata
	var cal: Dictionary = b.meta.get("motor", {})
	var fwd_drive := clampf(((motor["fwd_L"] + motor["fwd_R"]) * 0.5 - cal.get("fwd_offset_hz", 4.0)) / cal.get("fwd_full_hz", 50.0), 0.0, 1.0)
	var back_drive := clampf((motor["back"] - 4.0) / cal.get("back_full_hz", 40.0), 0.0, 1.0)
	speed = (fwd_drive - 0.7 * back_drive) * MAX_SPEED
	turn_rate = clampf((motor["turn_L"] - motor["turn_R"]) / cal.get("turn_full_hz", 45.0), -1.0, 1.0) * MAX_TURN

	proboscis = lerpf(proboscis, 1.0 if motor["mn9"] > cal.get("mn9_on_hz", 15.0) else 0.0, 1.0 - exp(-dt / 0.1))
	if _prob:
		_prob.scale = Vector3(1, 1 + proboscis * 1.5, 1)
		_prob.rotation.x = -0.3 - proboscis * 0.6
	feeding = proboscis > 0.5 and sense.get("taste", "") == "food"
	if feeding:
		hunger = maxf(0.0, hunger - dt * 0.08)
		world.consume(global_position, dt)
	else:
		hunger = minf(1.0, hunger + dt * 0.004)

	rotation.y += turn_rate * dt
	global_position += -global_transform.basis.z * speed * dt
	_clamp_to_arena()

	_phase += absf(speed) * dt * 28.0 + absf(turn_rate) * dt * 3.0
	_animate_body(0.0)
	for leg in _legs:
		var off := 0.0 if leg.get_meta("tripod") == 0 else PI
		var sx: float = leg.get_meta("sx")
		leg.rotation.y = leg.get_meta("base") + sin(_phase + off) * 0.35 * sx
		leg.rotation.x = maxf(0.0, cos(_phase + off)) * 0.25


func _clamp_to_arena() -> void:
	var r: float = world.ARENA_R - 0.3
	var p := Vector2(global_position.x, global_position.z)
	if p.length() > r:
		p = p.normalized() * r
		global_position.x = p.x
		global_position.z = p.y
		if airborne <= 0.0:
			rotation.y += PI * 0.5 * (1 if randf() < 0.5 else -1)
