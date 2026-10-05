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
var _axis := {}               # leg -> dof -> [node, local axis]
var _rest := {}
var _sens := []               # [leg, kind, neuron index, param]


func setup(p_brain: FlyBrain, p_fly) -> String:
	brain = p_brain
	fly = p_fly
	if not FileAccess.file_exists("res://data/leg_motor_map.json"):
		return "data/leg_motor_map.json missing (tools/build_leg_map.py)"
	_map = JSON.parse_string(FileAccess.get_file_as_string("res://data/leg_motor_map.json"))["legs"]
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
		var sn: Dictionary = _map.get(leg, {}).get("sensory", {})
		for kind in sn:
			var ids: Array = sn[kind]
			for k in ids.size():
				# alternate flexion/extension tuning; preferred claw angles spread over the FTi range
				_sens.append([leg, kind, int(ids[k]), float(k) / maxf(ids.size() - 1, 1)])
	enabled = true
	return ""


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


func odometry() -> Array:
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
		var flex_tuned := (k % 2) == 0
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
