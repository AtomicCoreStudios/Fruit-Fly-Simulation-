class_name FlyTaste
extends Node
## Taste organs on the real body. Every FlyWire gustatory receptor neuron (data/taste_neurons.csv,
## tools/build_taste.py) belongs to a sensor node placed in Blender (tools/blender/place_taste.py):
## labellar bristles and pegs on c_haustellum, leg sensors on each tarsal segment, one pharynx
## sensor in the rostrum. Each frame a sensor that physically touches a patch reads its chemistry,
## and each GRN fires according to its modality's dose-response (data/grn_response.json).
## Bench mode (--taste_bench=Hz) reproduces Shiu et al. 2024 Fig. 1: their 21 "sugar" GRNs at a
## fixed Poisson rate with no other taste input.

const CONTACT_CM := 0.035          # a sensor touches the floor when within 0.35 mm of it

var brain: FlyBrain
var fly                            # Fly
var world                          # Main
var resp: Dictionary
var bench_hz := 0.0
var bench_set := "shiu"            # "shiu": Shiu et al.'s 21 IDs; "sugar": corrected sugar-only set (right)
var sense := {}                    # for HUD/log: active GRNs per modality, contacts
var _idx := PackedInt32Array()
var _rate := PackedFloat32Array()
var _mod: Array[String] = []
var _organ: Array[String] = []
var _node: Array[Node3D] = []
var _bench := {}
var sugar_release := 1.0          # presynaptic release gain of sugar GRNs (hunger / dopamine, Inagaki et al. 2012)
var _sugar_release_set := 1.0


func setup(p_brain: FlyBrain, p_fly, p_world) -> String:
	brain = p_brain
	fly = p_fly
	world = p_world
	resp = JSON.parse_string(FileAccess.get_file_as_string("res://data/grn_response.json"))
	var organs: Dictionary = JSON.parse_string(FileAccess.get_file_as_string("res://data/taste_organs.json"))
	var f := FileAccess.open("res://data/taste_neurons.csv", FileAccess.READ)
	if f == null or resp.is_empty():
		return "missing taste data (run tools/build_taste.py)"
	var head := f.get_csv_line()
	var col := {}
	for i in head.size():
		col[head[i]] = i
	var shiu := {}
	for r in organs.get("shiu_sugar_R", []):
		shiu[str(r)] = true
	var missing := 0
	while not f.eof_reached():
		var row := f.get_csv_line()
		if row.size() < head.size():
			continue
		var sname: String = row[col["sensillum"]]
		var node: Node3D = fly.seg.get(sname)
		if node == null:
			missing += 1
			continue
		_idx.append(int(float(row[col["index"]])))
		_mod.append(row[col["modality"]])
		_organ.append(row[col["organ"]])
		_node.append(node)
		var rid := row[col["root_id"]]
		if shiu.has(rid):
			_bench["shiu"] = _bench.get("shiu", []) + [_idx.size() - 1]
		if row[col["organ"]] == "labellum_bristle" and row[col["side"]] == "right" and row[col["modality"]] in ["sugar", "sugar/low_salt"]:
			_bench["sugar"] = _bench.get("sugar", []) + [_idx.size() - 1]
		if row[col["organ"]] == "labellum_bristle" and row[col["side"]] == "left" and row[col["modality"]] in ["sugar", "sugar/low_salt"]:
			_bench["sugar_left"] = _bench.get("sugar_left", []) + [_idx.size() - 1]
		if row[col["organ"]] == "labellum_bristle" and row[col["modality"]] in ["sugar", "sugar/low_salt"]:
			_bench["sugar_both"] = _bench.get("sugar_both", []) + [_idx.size() - 1]
		if row[col["organ"]] == "leg_vnc" and row[col["modality"]] in ["sugar", "sugar/low_salt"]:
			_bench["leg_vnc_sugar"] = _bench.get("leg_vnc_sugar", []) + [_idx.size() - 1]
		if row[col["organ"]] == "leg" and row[col["modality"]] == "sugar":
			_bench["leg_sugar"] = _bench.get("leg_sugar", []) + [_idx.size() - 1]
		if row[col["organ"]] == "labellum_bristle" and row[col["modality"]] == "bitter":
			_bench["bitter"] = _bench.get("bitter", []) + [_idx.size() - 1]
	_rate.resize(_idx.size())
	print("taste: %d GRNs on %d sensor nodes (%d without a node); bench sets: %s" % [
		_idx.size(), _unique_nodes(), missing, str(_bench.keys().map(func(k): return "%s %d" % [k, _bench[k].size()]))])
	return ""


func _unique_nodes() -> int:
	var d := {}
	for n in _node:
		d[n] = true
	return d.size()


static func _hill(c: float, ec50: float, n: float) -> float:
	if c <= 0.0:
		return 0.0
	var x := pow(c / ec50, n)
	return x / (1.0 + x)


## Firing rate of one GRN of `modality` touching chemistry `chem` (empty = touching nothing).
func grn_rate(modality: String, chem: Dictionary) -> float:
	if chem.is_empty():
		return 0.0
	var m: Dictionary = resp["models"].get(modality, {})
	if m.is_empty():
		return 0.0                 # unknown / putative modalities: no known ligand in the arena
	var suc: float = chem.get("sucrose", 0.0)
	var na: float = chem.get("nacl", 0.0)
	var bit: float = chem.get("bitter", 0.0)
	match m["drive"]:
		"sucrose":
			var r: float = m["rmax"] * _hill(suc, m["ec50"], m["n"])
			if m.has("salt_rmax"):
				r += float(m["salt_rmax"]) * _hill(na, m["salt_ec50"], 1.0) * (1.0 if na < 150.0 else 150.0 / na)
			return r
		"water":
			if not chem.get("wet", false):
				return 0.0
			var osm := suc + 2.0 * na + bit
			return m["rmax"] / (1.0 + osm / float(m["osm_half"]))
		"nacl":
			return m["rmax"] * _hill(na, m["ec50"], m["n"])
		"bitter":
			return m["rmax"] * _hill(bit, m["ec50"], m["n"])
		"glutamate":
			return m["rmax"] * _hill(chem.get("glutamate", 0.0), m["ec50"], m["n"])
	return 0.0


var drive_idx := PackedInt32Array()     # diagnostic Poisson drive on arbitrary neurons (--drive=i,j:Hz)
var drive_rate := 0.0


func _send() -> void:
	var lg = fly.legs if fly != null else null
	if drive_idx.is_empty() and (lg == null or lg.idx.is_empty()):
		brain.set_sensor_rates(_idx, _rate)
		return
	var ii := _idx.duplicate()
	var rr := _rate.duplicate()
	if lg != null and bench_hz <= 0.0:          # leg proprioceptors (fly_legs.gd), not in taste benchmarks
		ii.append_array(lg.idx)
		rr.append_array(lg.rate)
	for i in drive_idx:
		ii.append(i)
		rr.append(drive_rate)
	brain.set_sensor_rates(ii, rr)


## Call every frame before brain.tick().
func update(_dt: float) -> void:
	if sugar_release != _sugar_release_set:
		# hunger-modulated release: sugar GRNs and the central nodes G2N-1 / Clavicle (data/hunger_targets.json)
		var si := PackedInt32Array()
		var ht = JSON.parse_string(FileAccess.get_file_as_string("res://data/hunger_targets.json")) if FileAccess.file_exists("res://data/hunger_targets.json") else null
		if ht != null:
			for i in ht["sugar_grn"]:
				si.append(int(i))
			for k in ht["central"]:
				for i in ht["central"][k]:
					si.append(int(i))
		else:
			for k in _idx.size():
				if _mod[k] in ["sugar", "sugar/low_salt"]:
					si.append(_idx[k])
		brain.set_release_gain(si, sugar_release)
		_sugar_release_set = sugar_release
	var counts := {}
	var contact := {"labellum": 0, "leg": 0}
	if bench_hz > 0.0:
		_rate.fill(0.0)
		for k in _bench.get(bench_set, []):
			_rate[k] = bench_hz
		_send()
		sense = {"bench": "%s set at %.0f Hz" % [bench_set, bench_hz]}
		return
	var cache := {}
	for k in _idx.size():
		var node := _node[k]
		var chem: Dictionary
		if cache.has(node):
			chem = cache[node]
		else:
			chem = {}
			if _organ[k] == "pharynx":
				if fly.feeding:
					chem = world.chemistry_at(fly.labellum_position())
			elif _organ[k].begins_with("labellum") and fly.airborne <= 0.0:
				# The NeuroMechFly neutral pose stands taller than a feeding fly (real flies lower
				# head and body to feed; posture is not simulated), so the labellum counts as on the
				# substrate when rostrum and haustellum are mostly extended. basis: approximate
				var reach: bool = fly.per_rostrum >= 0.7 and fly.per_haustellum >= 0.5
				if _organ[k] == "labellum_peg":
					reach = reach and fly.labellum_spread >= 0.3   # pegs face food only when the labellum opens (MN8)
				if reach:
					chem = world.chemistry_at(fly.labellum_position() * Vector3(1, 0, 1))
					if not chem.is_empty():
						contact["labellum"] += 1
			elif node.global_position.y < CONTACT_CM and fly.airborne <= 0.0:   # legs (brain-ascending and VNC-local GRNs)
				chem = world.chemistry_at(node.global_position)
				if not chem.is_empty():
					contact["leg" if _organ[k].begins_with("leg") else "labellum"] += 1
			cache[node] = chem
		var r := grn_rate(_mod[k], chem)
		_rate[k] = r
		if r > 1.0:
			counts[_mod[k]] = counts.get(_mod[k], 0) + 1
	_send()
	var low := INF
	for nd in _node:
		if nd.name.begins_with("leg_"):
			low = minf(low, nd.global_position.y)
	sense = {"active": counts, "contacts": contact, "lowest_tarsus_cm": snappedf(low, 0.0001)}
