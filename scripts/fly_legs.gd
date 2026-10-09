class_name FlyLegs
extends Node
## Neuromuscular legs: real leg motor neurons (BANC, data/leg_motor_map.json) -> muscle activation -> joint
## angles of the NeuroMechFly skeleton, and joint state -> the leg's real proprioceptor neurons.
##
## Motor: each motor neuron's rate r (from the brain's activity trace) sets a muscle activation
##   a = r / (r + R_HALF), low-passed with TAU_ACT. Per degree of freedom the summed agonist (pos) and antagonist
##   (neg) motor-unit activations set a torque balanced by passive joint stiffness: equilibrium
##   theta* = DTHETA_MN x (sum_pos - sum_neg), softly limited to the joint range (tanh), rest = 0; the joint
##   relaxes to it with TAU_JOINT (overdamped, passive forces dominate in small limbs). basis: motor-neuron -> muscle -> joint
##   assignment from BANC annotations (measured); activation and joint dynamics approximate.
## Sensory (Tuthill & Wilson 2016 review; Mamiya et al. 2018 Nat Neurosci for the FeCO):
##   FeCO claw  tonic position of the femur-tibia joint, half flexion- and half extension-tuned, preferred
##              angles spread over the joint range
##   FeCO hook  movement direction (half flexion-, half extension-selective)
##   FeCO club  bidirectional movement / vibration
##   hair plates  joint angle near the limits of the coxa and trochanter joints
##   campaniform sensilla  cuticular strain, here proportional to the leg's total muscle activation (load
##              proxy; ground load is added when walking)
## Rates are basis approximate (tens of Hz).

const LEGS := ["lf", "lm", "lh", "rf", "rm", "rh"]
const DOFS := ["ThC_pro", "ThC_add", "CTr", "FeRot", "FTi", "TiTa"]
const R_HALF := 30.0          # Hz for half activation
const TAU_ACT := 0.030        # s, muscle activation
const TAU_JOINT := 0.030      # s, joint relaxation (passive viscoelastic time constant, approximate)
const DTHETA_MN := 0.4        # rad of joint excursion per fully active motor neuron against passive stiffness
							  # (approximate calibration; motor-unit forces from Azevedo et al. 2020 eLife not
							  # yet converted to torque)
# rad, range reached at full activation of the pos / neg pool (approximate, NeuroMechFly kinematic ranges)
const RANGE := {"ThC_pro": [0.6, 0.6], "ThC_add": [0.3, 0.3], "CTr": [0.8, 0.8], "FeRot": [0.4, 0.0],
				"FTi": [1.2, 0.6], "TiTa": [0.5, 0.4]}

var brain: FlyBrain
var fly
var enabled := false
var angle := {}               # leg -> dof -> rad (relative to rest)
var omega := {}               # leg -> dof -> rad/s
var act := {}                 # motor neuron index -> activation
var sense := {}
var idx := PackedInt32Array() # proprioceptor neurons driven this frame
var rate := PackedFloat32Array()
var _map := {}
var _full := {}
var _presyn := {}             # movement afferent -> [[inhibitory presynaptic neuron, synapses], ...]
const I_HALF := 500.0         # synapse x Hz of presynaptic inhibition halving release (assumed)
var _axis := {}               # leg -> dof -> [node, local axis]
var _rest := {}
var _sens := []               # [leg, kind, neuron index, param]


func setup(p_brain: FlyBrain, p_fly) -> String:
	brain = p_brain
	fly = p_fly
	if not FileAccess.file_exists("res://data/leg_motor_map.json"):
		return "data/leg_motor_map.json missing (tools/build_leg_map.py)"
	_full = JSON.parse_string(FileAccess.get_file_as_string("res://data/leg_motor_map.json"))
	_map = _full["legs"]
	_presyn = _full.get("presyn_inhibition", {})
	for leg in LEGS:
		var c: Node3D = fly.seg.get(leg + "_coxa")
		var f: Node3D = fly.seg.get(leg + "_trochanterfemur")
		var t: Node3D = fly.seg.get(leg + "_tibia")
		var ta: Node3D = fly.seg.get(leg + "_tarsus1")
		var tip: Node3D = fly.seg.get(leg + "_tarsus5")
		if c == null or f == null or t == null or ta == null or tip == null:
			return "leg nodes missing for " + leg
		for nd in [c, f, t, ta]:
			_rest[nd] = nd.basis
		# flexion axis: normal of the plane of femur, tibia and tarsus at rest (world), expressed per parent
		var pf := f.global_position
		var pt := t.global_position
		var pa := ta.global_position
		var n := (pt - pf).cross(pa - pt)
		if n.length() < 1e-6:
			n = (pt - pf).cross(Vector3.UP)
		n = n.normalized()
		var up: Vector3 = fly.global_transform.basis.y
		var fwd: Vector3 = -fly.global_transform.basis.z
		_axis[leg] = {
			"ThC_pro": [c, _local_axis(c, up)], "ThC_add": [c, _local_axis(c, fwd)],
			"CTr": [f, _local_axis(f, n)], "FeRot": [f, _local_axis(f, (pt - pf).normalized())],
			"FTi": [t, _local_axis(t, n)], "TiTa": [ta, _local_axis(ta, n)]}
		# signs: + must protract (tip forward), adduct (tip medial/down), flex (fold tibia toward femur)
		_fix_sign(leg, "ThC_pro", tip, func(p0, p1): return (p1 - p0).dot(fwd))
		var lat: Vector3 = (c.global_position - fly.global_position)
		lat.y = 0.0
		lat = lat.normalized()
		_fix_sign(leg, "ThC_add", tip, func(p0, p1): return -(p1 - p0).dot(lat) - (p1 - p0).dot(up))
		_fix_sign(leg, "CTr", tip, func(p0, p1): return -(p1 - c.global_position).length() + (p0 - c.global_position).length())
		_fix_sign(leg, "FTi", tip, func(p0, p1): return -(p1 - pf).length() + (p0 - pf).length())
		_fix_sign(leg, "TiTa", tip, func(p0, p1): return -(p1 - up * 0.0).dot(up) + p0.dot(up))
		angle[leg] = {}
		omega[leg] = {}
		for d in DOFS:
			angle[leg][d] = 0.0
			omega[leg][d] = 0.0
		var tun: Dictionary = _full.get("feco_tuning", {})
		var sn: Dictionary = _map.get(leg, {}).get("sensory", {})
		for kind in sn:
			var ids: Array = sn[kind]
			for k in ids.size():
				# alternate flexion/extension tuning; preferred claw angles spread over the FTi range
				# 5th field: flexion/extension tuning of claw/hook neurons from their wiring (Lee et al. 2025 criterion;
				# tools/build_leg_map.py), "" for other kinds
				_sens.append([leg, kind, int(ids[k]), float(k) / maxf(ids.size() - 1, 1), str(tun.get(str(int(ids[k])), ""))])
	enabled = true
	if _user_arg("physics", "") == "mujoco":
		enabled = false
		return mujoco_start()
	if "--legs_lift_test" in OS.get_cmdline_user_args():
		_lift_test()
	return ""


func _lift_test() -> void:
	# foot height change (fraction of reach, body frame) for +0.3 rad of each DoF from rest (+ = protract,
	# adduct, flex, rotate, flex, depress per the sign convention); positive = foot rises
	var inv: Transform3D = fly.global_transform.affine_inverse()
	for leg in LEGS:
		var tip: Node3D = fly.seg.get(leg + "_tarsus5")
		var p0: Vector3 = inv * tip.global_position
		var reach := p0.length()
		var out := []
		for d in DOFS:
			angle[leg][d] = 0.3
			_apply(leg)
			var p1: Vector3 = inv * tip.global_position
			out.append("%s %+.2f" % [d, (p1.y - p0.y) / reach])
			angle[leg][d] = 0.0
			_apply(leg)
		print("lift_test ", leg, ": ", ", ".join(out))


## Leg-driven locomotion (no-slip stance): feet whose height in the body frame is below their rest height
## (plus STANCE_TOL of the leg's reach) grip the ground; the body moves by minus the summed stance-foot
## motion (translation in the horizontal body plane and yaw about the body centre). Returns
## [forward cm, lateral cm, yaw rad] for this frame. basis: kinematic contact model (approximate; no physics
## engine forces, no slipping or body pitch).
const STANCE_TOL := 0.05
const LOAD_HZ := 40.0         # campaniform rate for a leg carrying a third of body weight (tripod), approximate
var _foot_prev := {}
var _foot_rest := {}
var stance := {}
var foot_height := {}         # foot height above its rest height, as a fraction of leg reach


func odometry() -> Array:
	if physics_mujoco:
		return mujoco_odometry()
	var inv: Transform3D = fly.global_transform.affine_inverse()
	var dp := Vector3.ZERO
	var dyaw := 0.0
	var ns := 0
	for leg in LEGS:
		var tip: Node3D = fly.seg.get(leg + "_tarsus5")
		var p: Vector3 = inv * tip.global_position
		if not _foot_rest.has(leg):
			_foot_rest[leg] = p
			_foot_prev[leg] = p
		var reach: float = (_foot_rest[leg] as Vector3).length()
		foot_height[leg] = (p.y - (_foot_rest[leg] as Vector3).y) / maxf(reach, 1e-6)
		var on: bool = p.y <= (_foot_rest[leg] as Vector3).y + STANCE_TOL * reach
		stance[leg] = on
		if on:
			var d: Vector3 = p - _foot_prev[leg]
			d.y = 0.0
			dp += d
			var r := Vector3(p.x, 0, p.z)
			if r.length() > 1e-5:
				dyaw += r.cross(d).y / r.length_squared()
			ns += 1
		_foot_prev[leg] = p
	if ns == 0:
		return [0.0, 0.0, 0.0]
	dp /= float(ns)
	# feet moving backward (+z in body frame) push the body forward (-z)
	return [dp.z, -dp.x, -dyaw / float(ns)]


func _local_axis(node: Node3D, world_axis: Vector3) -> Vector3:
	return (node.get_parent() as Node3D).global_transform.basis.inverse() * world_axis


func _fix_sign(leg: String, dof: String, tip: Node3D, metric: Callable) -> void:
	var nd: Node3D = _axis[leg][dof][0]
	var ax: Vector3 = _axis[leg][dof][1].normalized()
	var p0 := tip.global_position
	nd.basis = _rest[nd] * Basis(_rest[nd].inverse() * ax, 0.05)
	var p1 := tip.global_position
	nd.basis = _rest[nd]
	if metric.call(p0, p1) < 0.0:
		_axis[leg][dof][1] = -ax


## Call every frame after brain.tick().
func update(dt: float) -> void:
	if not enabled:
		return
	if physics_mujoco:
		_mujoco_update()
		return
	var a := brain.activity_bytes.to_float32_array()
	var to_hz := 1000.0 / brain.act_tau_ms
	var ka := 1.0 - exp(-dt / TAU_ACT)
	var kj := 1.0 - exp(-dt / TAU_JOINT)
	var load := {}
	for leg in LEGS:
		var mot: Dictionary = _map.get(leg, {}).get("motor", {})
		var tot := 0.0
		for d in DOFS:
			var A := [0.0, 0.0]
			if mot.has(d):
				for s in 2:
					var key: String = ["pos", "neg"][s]
					var ids: Array = mot[d][key]
					var ws: Array = mot[d]["w_" + key]
					var sw := 0.0
					for k in ids.size():
						var i := int(ids[k])
						var r := maxf(a[i] * to_hz, 0.0) if i < a.size() else 0.0
						var x: float = act.get(i, 0.0)
						x += ((r / (r + R_HALF)) - x) * ka
						act[i] = x
						A[s] += x * float(ws[k])   # motor-unit forces add (summed, not averaged)
			tot += A[0] + A[1]
			# torque balance against passive stiffness (overdamped joint, passive forces dominate in small
			# limbs: Hooper et al. 2009): equilibrium = DTHETA_MN x (summed agonist - antagonist activation),
			# softly limited to the joint range; passive torque returns the joint to rest when MNs pause
			var eq: float = DTHETA_MN * (A[0] - A[1])
			var lim: float = RANGE[d][0] if eq >= 0.0 else RANGE[d][1]
			var target: float = lim * tanh(eq / maxf(lim, 1e-3)) if lim > 0.0 else 0.0
			var old: float = angle[leg][d]
			var nw := old + (target - old) * kj
			angle[leg][d] = nw
			omega[leg][d] = (nw - old) / maxf(dt, 1e-4)
		load[leg] = tot
		_apply(leg)
	_proprio(load)


func _apply(leg: String) -> void:
	var per_node := {}
	for d in DOFS:
		var nd: Node3D = _axis[leg][d][0]
		var ax: Vector3 = _axis[leg][d][1]
		if not per_node.has(nd):
			per_node[nd] = Basis()
		per_node[nd] = per_node[nd] * Basis((_rest[nd].inverse() * ax).normalized(), angle[leg][d])
	for nd in per_node:
		nd.basis = _rest[nd] * per_node[nd]


func _proprio(load: Dictionary) -> void:
	idx.resize(_sens.size())
	rate.resize(_sens.size())
	var hp := {}
	for k in _sens.size():
		var leg: String = _sens[k][0]
		var kind: String = _sens[k][1]
		var u: float = _sens[k][3]
		var tlab: String = _sens[k][4]
		var flex_tuned := (tlab == "flex") if tlab != "" else ((k % 2) == 0)
		var th: float = angle[leg]["FTi"]
		var w: float = omega[leg]["FTi"]
		var r := 0.0
		match kind:
			"FeCO_claw":
				var pref := lerpf(-0.6, 1.2, u)
				var s := 1.0 if flex_tuned else -1.0
				r = 60.0 / (1.0 + exp(-s * (th - pref) / 0.1))
			"FeCO_hook":
				r = clampf((w if flex_tuned else -w) * 20.0, 0.0, 120.0)
			"FeCO_club":
				r = clampf(absf(w) * 10.0, 0.0, 120.0)
			"hair_plate":
				var d := "ThC_pro" if flex_tuned else "CTr"
				var x: float = absf(angle[leg][d]) / RANGE[d][0]
				r = 80.0 / (1.0 + exp(-(x - 0.6) / 0.08))
			"campaniform":
				# cuticular strain: own muscle force + ground load while the foot is in stance (body weight shared
				# by the stance legs; Zill et al. 2004 review: campaniform sensilla encode leg load). approximate
				var ns := 0
				for l2 in stance:
					ns += 1 if stance[l2] else 0
				var ground := (LOAD_HZ / float(maxi(ns, 1))) * 3.0 if stance.get(leg, false) else 0.0
				r = clampf(load.get(leg, 0.0) * 25.0 + ground, 0.0, 150.0)
		idx[k] = _sens[k][2]
		rate[k] = r
		hp[kind] = hp.get(kind, 0.0) + r / _sens.size()
	sense = hp
	_presyn_gate()


func _presyn_gate() -> void:
	# Dallmann et al. 2025: GABAergic presynaptic inhibition suppresses movement-encoding (hook, club) afferents
	# during self-generated movement. Release gain = 1 / (1 + sum(syn x rate of inhibitory inputs) / I_HALF).
	if _presyn.is_empty():
		return
	var a := brain.activity_bytes.to_float32_array()
	var to_hz := 1000.0 / brain.act_tau_ms
	var gsum := 0.0
	var gn := 0
	for key in _presyn:
		var inh := 0.0
		for pr in _presyn[key]:
			var j := int(pr[0])
			if j < a.size():
				inh += float(pr[1]) * maxf(a[j] * to_hz, 0.0)
		var gain := 1.0 / (1.0 + inh / I_HALF)
		brain.set_release_gain(PackedInt32Array([int(key)]), snappedf(gain, 0.02))
		gsum += gain
		gn += 1
	sense["presyn_gain_mean"] = gsum / maxf(gn, 1)


# ---------------- Physical body in MuJoCo (--physics=mujoco) ----------------
## The body is simulated in MuJoCo by tools/body_server.py (FlyGym NeuroMechFly, true scale, contact physics, pad
## adhesion). Each motor update sends the brain's real leg motor-neuron rates and receives every segment's pose and
## the rates of the leg's real proprioceptors. Every segment node (including c_head, which carries the compound-eye
## cameras and facets) takes the MuJoCo pose, so the eyes sit on the physical head. Physics time = brain simulated
## time. Not driven by MuJoCo yet: proboscis (c_rostrum, c_haustellum; PER animation) and wings (flight not modelled).
## Started with the project's .venv-flygym; the server listens on 127.0.0.1 only and stops with Godot.
const MJ_PORT := 47830
const MJ_SKIP := ["c_rostrum", "c_haustellum", "l_wing", "r_wing"]
const _C := Basis(Vector3(1, 0, 0), Vector3(0, 0, -1), Vector3(0, 1, 0))   # MuJoCo/Blender z-up -> glTF/Godot y-up
const THORAX_REST_X := 0.496           # mm, thorax origin in the rig frame (data/flygym_rig.json)
var physics_mujoco := false
var mj_info := {}
var _tcp: StreamPeerTCP
var _srv_pid := -1
var _mj_mn := PackedInt32Array()
var _mj_nodes: Array = []
var _mj_parent := PackedInt32Array()
var _mj_thorax := -1
var _mj_ns := 0
var _mj_prev := Vector3.ZERO           # virtual root (x, y, yaw) in the MuJoCo world
var _mj_have_prev := false
var _mj_odo := [0.0, 0.0, 0.0]
var _mj_last_ms := -1.0
var mj_min_z := 99.0                   # lowest thorax height (mm) after 2 s (upright check for warm-ups)
var mj_min_up := 1.0                   # lowest z-component of the thorax up axis after 2 s
var head_yaw := 0.0                    # rad, + = left (MuJoCo neck, driven by the real neck motor neurons)
var head_roll := 0.0


static func _user_arg(key: String, def: String) -> String:
	for a in OS.get_cmdline_user_args():
		if a.begins_with("--" + key + "="):
			return a.split("=", true, 1)[1]
	return def


func mujoco_start() -> String:
	var py := ProjectSettings.globalize_path("res://.venv-flygym/Scripts/python.exe")
	if not FileAccess.file_exists(py):
		return ".venv-flygym missing (README: FlyGym environment)"
	var port := int(_user_arg("mj_port", str(MJ_PORT)))
	var argv := [ProjectSettings.globalize_path("res://tools/body_server.py"), "--port", str(port),
		"--pad_fmax", _user_arg("pad_fmax", "10"), "--tether", _user_arg("mj_tether", "0")]
	_srv_pid = OS.create_process(py, argv, false)
	if _srv_pid <= 0:
		return "could not start tools/body_server.py"
	_tcp = StreamPeerTCP.new()
	var t0 := Time.get_ticks_msec()
	while Time.get_ticks_msec() - t0 < 120000:
		var st := _tcp.get_status()
		if st == StreamPeerTCP.STATUS_NONE or st == StreamPeerTCP.STATUS_ERROR:
			_tcp = StreamPeerTCP.new()
			_tcp.connect_to_host("127.0.0.1", port)
		_tcp.poll()
		if _tcp.get_status() == StreamPeerTCP.STATUS_CONNECTED:
			break
		OS.delay_msec(250)
	if _tcp.get_status() != StreamPeerTCP.STATUS_CONNECTED:
		return "body server did not accept a connection within 120 s"
	_tcp.set_no_delay(true)
	var line := PackedByteArray()
	while true:
		var b := _mj_read(1)
		if b.is_empty():
			return "body server handshake timed out"
		if b[0] == 10:
			break
		line.append(b[0])
	mj_info = JSON.parse_string(line.get_string_from_utf8())
	_mj_mn = PackedInt32Array(mj_info["mn_ids"])
	idx = PackedInt32Array(mj_info["sensor_ids"])
	_mj_ns = idx.size()
	rate.resize(_mj_ns)
	var names: Array = mj_info["segments"]
	# --mj_rigid_head=1 (diagnostic): the head keeps its rest pose on the thorax (neck motion not shown to the eyes)
	var skip: Array = MJ_SKIP.duplicate()
	if _user_arg("mj_rigid_head", "0") == "1":
		skip.append("c_head")
	for nm in names:
		_mj_nodes.append(null if skip.has(nm) else fly.seg.get(nm))
	for k in names.size():
		var nd: Node3D = fly.seg.get(names[k])
		var p := -1
		if nd != null and nd.get_parent() != null:
			p = names.find(str(nd.get_parent().name))
		_mj_parent.append(p)
	_mj_thorax = names.find("c_thorax")
	# self-check: the MuJoCo neutral pose (handshake) mapped exactly as in _mujoco_update must reproduce every Godot
	# segment's rest transform (built in Blender from the same rig and neutral pose); prints the largest deviation
	var rp: Array = mj_info["rest_poses"]
	var RT: Array[Transform3D] = []
	for r in rp:
		RT.append(Transform3D(Basis(Quaternion(r[4], r[5], r[6], r[3])), Vector3(r[0], r[1], r[2])))
	var worst := 0.0
	var worst_info := ""
	for k in names.size():
		var nd: Node3D = fly.seg.get(names[k])
		if nd == null:
			continue
		var rel: Transform3D = RT[k] if _mj_parent[k] < 0 else RT[_mj_parent[k]].affine_inverse() * RT[k]
		var g := Transform3D(_C * rel.basis * _C.transposed(), _C * rel.origin)
		if _mj_parent[k] < 0:
			g.origin = nd.position       # root height/position are set by the physics, compare orientation only
		var dev := maxf((nd.position - g.origin).length(), (nd.basis.x - g.basis.x).length() + (nd.basis.y - g.basis.y).length() + (nd.basis.z - g.basis.z).length())
		if dev > worst:
			worst = dev
			worst_info = "%s: node %s | mapped MuJoCo %s" % [names[k], str(nd.transform), str(g)]
	physics_mujoco = true
	enabled = true
	print("legs: MuJoCo body connected (%d segments, %d motor neurons, %d proprioceptors; pose-mapping check max deviation %.6f)"
		% [names.size(), _mj_mn.size(), _mj_ns, worst])
	if worst > 1e-3:
		print("legs: pose-mapping check worst segment ", worst_info)
	return ""


func _mj_read(n: int) -> PackedByteArray:
	var t0 := Time.get_ticks_msec()
	while _tcp.get_available_bytes() < n:
		_tcp.poll()
		if _tcp.get_status() != StreamPeerTCP.STATUS_CONNECTED or Time.get_ticks_msec() - t0 > 60000:
			return PackedByteArray()
		OS.delay_usec(100)
	var r: Array = _tcp.get_data(n)
	return r[1] if r[0] == OK else PackedByteArray()


func _mujoco_update() -> void:
	# physics advances by the brain's simulated time since the last motor update
	var now_ms: float = brain.sim_time_ms
	var dts := 0.0 if _mj_last_ms < 0.0 else (now_ms - _mj_last_ms) / 1000.0
	_mj_last_ms = now_ms
	if dts <= 0.0:
		return
	var a := brain.activity_bytes.to_float32_array()
	var to_hz := 1000.0 / brain.act_tau_ms
	var msg := PackedFloat32Array()
	msg.resize(1 + _mj_mn.size())
	msg[0] = dts
	for k in _mj_mn.size():
		var i := _mj_mn[k]
		msg[k + 1] = maxf(a[i] * to_hz, 0.0) if i < a.size() else 0.0
	var head := PackedByteArray()
	head.resize(4)
	head.encode_u32(0, msg.size())
	_tcp.put_data(head + msg.to_byte_array())
	var hb := _mj_read(4)
	if hb.size() < 4:
		push_error("body server stopped answering")
		enabled = false
		return
	var out := _mj_read(4 * hb.decode_u32(0)).to_float32_array()
	var n := _mj_nodes.size()
	var T: Array[Transform3D] = []
	for s in n:
		var o := s * 7
		T.append(Transform3D(Basis(Quaternion(out[o + 4], out[o + 5], out[o + 6], out[o + 3])), Vector3(out[o], out[o + 1], out[o + 2])))
	# heading and planar position move the fly node (odometry); everything else is shown on the segment nodes
	var th := T[_mj_thorax]
	var yaw := atan2(th.basis.x.y, th.basis.x.x)
	var root := Vector2(th.origin.x, th.origin.y) - Vector2(cos(yaw), sin(yaw)) * THORAX_REST_X
	if _mj_have_prev:
		var d := (root - Vector2(_mj_prev.x, _mj_prev.y)).rotated(-_mj_prev.z)
		_mj_odo[0] += d.x * 0.1            # mm -> arena units (cm); + = forward
		_mj_odo[1] += -d.y * 0.1           # MuJoCo +y is the fly's left; odometry + = right
		_mj_odo[2] += wrapf(yaw - _mj_prev.z, -PI, PI)
	_mj_prev = Vector3(root.x, root.y, yaw)
	_mj_have_prev = true
	var root_inv := Transform3D(Basis(Vector3(0, 0, 1), yaw), Vector3(root.x, root.y, 0.0)).affine_inverse()
	for s in n:
		var nd: Node3D = _mj_nodes[s]
		if nd == null:
			continue
		var rel: Transform3D = (root_inv if _mj_parent[s] < 0 else T[_mj_parent[s]].affine_inverse()) * T[s]
		nd.transform = Transform3D(_C * rel.basis * _C.transposed(), _C * rel.origin)
	var o2 := 7 * n
	for k in _mj_ns:
		rate[k] = out[o2 + k]
	o2 += _mj_ns
	for li in LEGS.size():
		var leg: String = LEGS[li]
		stance[leg] = out[o2 + li] > 0.5
		foot_height[leg] = out[o2 + 18 + li]
		for di in DOFS.size():
			angle[leg][DOFS[di]] = out[o2 + 24 + li * 6 + di]
	sense = {"thorax_z_mm": th.origin.z, "thorax_up": th.basis.z.z, "load_lf": out[o2 + 6], "load_rf": out[o2 + 9]}
	if brain.sim_time_ms > 2000.0:
		mj_min_z = minf(mj_min_z, th.origin.z)
		mj_min_up = minf(mj_min_up, th.basis.z.z)
	if out.size() >= o2 + 24 + 36 + 2:       # protocol v2: head yaw/roll (rad, + = left) driven by the neck motor neurons
		head_yaw = out[o2 + 60]
		head_roll = out[o2 + 61]
		sense["head_yaw"] = head_yaw
		sense["head_roll"] = head_roll
	_presyn_gate()


func mujoco_odometry() -> Array:
	var od := _mj_odo.duplicate()
	_mj_odo = [0.0, 0.0, 0.0]
	return od


func _exit_tree() -> void:
	if _tcp != null and _tcp.get_status() == StreamPeerTCP.STATUS_CONNECTED:
		var z := PackedByteArray()
		z.resize(4)
		z.encode_u32(0, 0)
		_tcp.put_data(z)
		_tcp.disconnect_from_host()
	if _srv_pid > 0 and OS.is_process_running(_srv_pid):
		OS.delay_msec(300)
		if OS.is_process_running(_srv_pid):
			OS.kill(_srv_pid)
