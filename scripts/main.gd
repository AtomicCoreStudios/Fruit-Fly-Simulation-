extends Node3D
## FLY-UI: an embodied whole-brain emulation of Drosophila.
## Builds the arena, the fly, the GPU brain, the HUD and the live brain view.
##
## Command-line (after `--`):  --log  --auto  --frames=N  --shot=screenshots/x.png

const ARENA_R := 12.0
const ODOUR_SIGMA := 3.5

var brain := FlyBrain.new()
var fly: Fly
var sun_azimuth := 0.6
var wind := Vector3.ZERO
var patches: Array = []       # {pos, radius, kind, amount, node}
var pillars: Array = []
var threat: MeshInstance3D
var threat_vel := Vector3.ZERO

var cam: Camera3D
var cam_mode := 0
var sun: DirectionalLight3D
var brain_cam: Camera3D
var brain_mm: MultiMeshInstance3D
var act_tex: ImageTexture
var brain_view: Control
var hud_stats: Label
var hud_regions: Bars
var hud_io: Bars
var hud_ring: Ring
var hud_msg: Label
var error_text := ""

var _args := {}
var _frame := 0
var _log_t := 0.0
var _orbit := 0.0


# =================================================================== setup
func _ready() -> void:
	for a in OS.get_cmdline_user_args():
		var kv := a.trim_prefix("--").split("=", true, 1)
		_args[kv[0]] = kv[1] if kv.size() > 1 else "1"
	_build_world()
	var which: String = _args.get("connectome", "flywire")
	if not FileAccess.file_exists("res://data/connectome_%s.bin" % which):
		which = "synthetic"
	brain.physiology = _args.get("physiology", "1") != "0"
	brain.homeostasis_learning = _args.has("homeostasis")
	brain.conductance_synapses = _args.get("synapses", "conductance") != "current"
	brain.act_tau_ms = float(_args.get("act_tau", "60"))
	if _args.has("phys_ablate"):
		brain.phys_ablate = Array(str(_args["phys_ablate"]).split(","))
	var err := brain.load_connectome("res://data/connectome_%s.bin" % which, "res://data/connectome_%s.json" % which)
	_build_hud()
	if err != "":
		error_text = err
		hud_msg.text = "BRAIN NOT LOADED: " + err
		printerr(err)
		return
	fly = Fly.new()
	fly.brain = brain
	fly.world = self
	fly.vision_mode = _args.get("vision", "facets")
	fly.lamina_bias_mv = float(_args.get("lamina_bias", "9.0"))
	fly.columnar_bias_mv = float(_args.get("columnar_bias", "0.0"))
	fly.tethered = _args.has("tethered")
	fly.optic_lobe = _args.get("optic_lobe", "flyvis")
	fly.flyvis_gain = float(_args.get("flyvis_gain", "60"))
	fly.graded_vpn = _args.get("graded", "1") != "0"
	fly.graded_c_mv = float(_args.get("vpn_c", "19"))
	fly.taste_bench_hz = float(_args.get("taste_bench", "0"))
	fly.taste_bench_set = _args.get("taste_set", "shiu")
	fly.clean_air = _args.has("clean_air")
	fly.drive_spec = str(_args.get("drive", ""))
	fly.sugar_release_override = float(_args.get("sugar_release", "0"))
	if _args.has("silence"):     # diagnostic ablation: model indices whose transmitter release is set to 0
		var si := PackedInt32Array()
		for x in str(_args["silence"]).split(","):
			si.append(int(x))
		brain.set_release_gain(si, 0.0)
	if _args.has("release_gain"):   # diagnostic: "i,j,k:gain;l,m:gain" sets presynaptic release gains
		for part in str(_args["release_gain"]).split(";"):
			var kv := part.split(":")
			var ri := PackedInt32Array()
			for x in kv[0].split(","):
				ri.append(int(x))
			brain.set_release_gain(ri, float(kv[1]))
	if _args.has("opto") and _args["opto"] != "1":
		for kv in str(_args["opto"]).split(","):
			var parts := kv.split(":")
			fly.opto[parts[0]] = float(parts[1]) if parts.size() > 1 else 100.0
	add_child(fly)
	fly.position = Vector3(-2, 0.06, 7)
	if _args.has("start"):
		var xz: PackedStringArray = _args["start"].split(",")
		fly.position = Vector3(float(xz[0]), 0.06, float(xz[1]))
	if _args.has("yaw"):
		fly.rotation.y = float(_args["yaw"])
	_build_brain_view()
	print("FLY-UI: %d neurons, %d synaptic edges (%s); intrinsic physiology %s" % [brain.n, brain.e, brain.meta["source"],
		("ON (spontaneous activity + adaptation" + (", homeostasis LEARNING" if brain.homeostasis_learning else (", homeostatic offsets loaded" if brain.homeostasis_loaded else "")) + ")") if brain.physiology_loaded else "off"])
	if brain.flyvis_enabled:
		print("optic lobe: FlyVis %s graded model, %d cells x 2 eyes driving %d FlyWire neurons (gain %.0f Hz)" % [
			brain.flyvis_meta["model"], brain.flyvis_meta["n_nodes"], brain.flyvis_meta["n_out"], brain.flyvis_gain])
	if brain.graded_enabled:
		print("graded optic lobe + VPN dendrites: %d units, %d synaptic edges, VPN soma coupling %.1f mV/unit" % [
			brain.graded_meta["n_g"], brain.graded_meta["n_edges"], brain.graded_c_mv])


func _build_world() -> void:
	var env := Environment.new()
	env.background_mode = Environment.BG_SKY
	var sky := Sky.new()
	var sm := ProceduralSkyMaterial.new()
	sm.sky_top_color = Color(0.35, 0.55, 0.85)
	sm.sky_horizon_color = Color(0.75, 0.8, 0.88)
	sm.ground_bottom_color = Color(0.2, 0.18, 0.15)
	sky.sky_material = sm
	env.sky = sky
	env.ambient_light_source = Environment.AMBIENT_SOURCE_SKY
	env.tonemap_mode = Environment.TONE_MAPPER_ACES
	env.tonemap_exposure = 0.85
	env.ambient_light_energy = 0.6
	var we := WorldEnvironment.new()
	we.environment = env
	add_child(we)

	sun = DirectionalLight3D.new()
	sun.shadow_enabled = true
	sun.light_energy = 0.9
	add_child(sun)
	_update_sun()

	# arena floor: checker disc
	var img := Image.create(256, 256, false, Image.FORMAT_RGB8)
	for y in 256:
		for x in 256:
			var c := 0.52 if ((x / 32) + (y / 32)) % 2 == 0 else 0.44
			img.set_pixel(x, y, Color(c, c * 0.97, c * 0.9))
	var fm := StandardMaterial3D.new()
	fm.albedo_texture = ImageTexture.create_from_image(img)
	fm.uv1_scale = Vector3(6, 6, 1)
	fm.roughness = 0.9
	var floor_mi := MeshInstance3D.new()
	var cyl := CylinderMesh.new()
	cyl.top_radius = ARENA_R
	cyl.bottom_radius = ARENA_R
	cyl.height = 0.1
	cyl.radial_segments = 96
	floor_mi.mesh = cyl
	floor_mi.material_override = fm
	floor_mi.position.y = -0.05
	add_child(floor_mi)
	var wall := MeshInstance3D.new()
	var tm := TorusMesh.new()
	tm.inner_radius = ARENA_R
	tm.outer_radius = ARENA_R + 0.25
	tm.rings = 96
	wall.mesh = tm
	var wm := StandardMaterial3D.new()
	wm.albedo_color = Color(0.9, 0.9, 0.92)
	wall.material_override = wm
	add_child(wall)

	# dark landmark pillars (visual objects the fly can fixate)
	var pm := StandardMaterial3D.new()
	pm.albedo_color = Color(0.08, 0.08, 0.1)
	for i in 3:
		var a := TAU * i / 3.0 + 0.4
		var p := Vector3(cos(a) * 8.5, 1.5, sin(a) * 8.5)
		var mi := MeshInstance3D.new()
		var c := CylinderMesh.new()
		c.top_radius = 0.45
		c.bottom_radius = 0.45
		c.height = 3.0
		mi.mesh = c
		mi.material_override = pm
		mi.position = p
		add_child(mi)
		pillars.append({"pos": p, "radius": 0.45, "height": 3.0, "darkness": 0.9})

	add_patch(Vector3(-4, 0, -3), "food")
	add_patch(Vector3(5, 0, -5), "food")
	add_patch(Vector3(3, 0, 4), "aversive")

	threat = MeshInstance3D.new()
	var ts := SphereMesh.new()
	ts.radius = 0.7
	ts.height = 1.4
	threat.mesh = ts
	var thm := StandardMaterial3D.new()
	thm.albedo_color = Color(0.05, 0.05, 0.05)
	threat.material_override = thm
	threat.visible = false
	add_child(threat)

	cam = Camera3D.new()
	cam.fov = 55
	add_child(cam)
	cam.position = Vector3(0, 6, 10)
	cam.look_at(Vector3.ZERO)


func add_patch(p: Vector3, kind: String) -> void:
	var mi := MeshInstance3D.new()
	var c := CylinderMesh.new()
	c.top_radius = 0.9
	c.bottom_radius = 0.9
	c.height = 0.04
	mi.mesh = c
	var m := StandardMaterial3D.new()
	m.albedo_color = Color(1.0, 0.8, 0.15) if kind == "food" else Color(0.55, 0.15, 0.7)
	m.emission_enabled = true
	m.emission = m.albedo_color * 0.3
	mi.material_override = m
	mi.position = Vector3(p.x, 0.02, p.z)
	add_child(mi)
	patches.append({"pos": Vector3(p.x, 0, p.z), "radius": 0.9, "kind": kind, "amount": 1.0, "node": mi})


func _update_sun() -> void:
	# sun_azimuth uses the same convention as a node's rotation.y (0 = -Z)
	# the light shines along its -Z: yaw it to point away from the sun, pitch down
	sun.rotation = Vector3(-0.8, sun_azimuth + PI, 0)


func _build_brain_view() -> void:
	var svc := SubViewportContainer.new()
	svc.stretch = true
	svc.anchor_left = 1
	svc.anchor_top = 1
	svc.anchor_right = 1
	svc.anchor_bottom = 1
	svc.offset_left = -560
	svc.offset_top = -420
	svc.offset_right = -10
	svc.offset_bottom = -10
	$HUD.add_child(svc)
	brain_view = svc
	var sv := SubViewport.new()
	sv.own_world_3d = true
	sv.msaa_3d = Viewport.MSAA_2X
	svc.add_child(sv)
	var env := Environment.new()
	env.background_mode = Environment.BG_COLOR
	env.background_color = Color(0.015, 0.015, 0.03)
	env.glow_enabled = true
	env.glow_intensity = 0.5
	env.glow_bloom = 0.0
	env.glow_hdr_threshold = 0.9
	var we := WorldEnvironment.new()
	we.environment = env
	sv.add_child(we)
	brain_cam = Camera3D.new()
	brain_cam.fov = 45
	sv.add_child(brain_cam)

	var mm := MultiMesh.new()
	mm.transform_format = MultiMesh.TRANSFORM_3D
	mm.use_colors = true
	mm.mesh = QuadMesh.new()
	mm.instance_count = brain.n
	mm.buffer = brain.multimesh_buffer
	brain_mm = MultiMeshInstance3D.new()
	brain_mm.multimesh = mm
	var mat := ShaderMaterial.new()
	mat.shader = load("res://shaders/neuron_view.gdshader")
	var img := Image.create_from_data(brain.meta["tex_width"], brain.n_pad / int(brain.meta["tex_width"]),
		false, Image.FORMAT_RF, brain.activity_bytes)
	act_tex = ImageTexture.create_from_image(img)
	mat.set_shader_parameter("activity", act_tex)
	mat.set_shader_parameter("tex_width", int(brain.meta["tex_width"]))
	brain_mm.material_override = mat
	brain_mm.extra_cull_margin = 16.0
	sv.add_child(brain_mm)

	var cap := Label.new()
	cap.text = "LIVE CONNECTOME  ·  %s neurons  ·  frontal orbit" % _thousands(brain.n)
	cap.position = Vector2(10, 6)
	cap.add_theme_font_size_override("font_size", 13)
	cap.add_theme_color_override("font_color", Color(0.7, 0.85, 1.0))
	svc.add_child(cap)


func _build_hud() -> void:
	var hud := CanvasLayer.new()
	hud.name = "HUD"
	add_child(hud)
	var bar := ColorRect.new()
	bar.color = Color(0.02, 0.03, 0.06, 0.78)
	bar.size = Vector2(760, 132)
	hud.add_child(bar)
	var help_bg := ColorRect.new()
	help_bg.color = Color(0.02, 0.03, 0.06, 0.7)
	help_bg.anchor_top = 1
	help_bg.anchor_bottom = 1
	help_bg.offset_top = -126
	help_bg.offset_bottom = -76
	help_bg.offset_right = 790
	hud.add_child(help_bg)
	var title := Label.new()
	title.text = "FLY-UI  ·  Uploaded Intelligence emulation  ·  Drosophila melanogaster"
	title.position = Vector2(14, 8)
	title.add_theme_font_size_override("font_size", 20)
	title.add_theme_color_override("font_color", Color(0.75, 0.9, 1.0))
	hud.add_child(title)
	hud_stats = Label.new()
	hud_stats.position = Vector2(14, 38)
	hud_stats.add_theme_font_size_override("font_size", 13)
	hud.add_child(hud_stats)
	hud_regions = Bars.new()
	hud_regions.position = Vector2(14, 150)
	hud_regions.heading = "NEUROPIL FIRING (Hz / neuron)"
	hud.add_child(hud_regions)
	hud_io = Bars.new()
	hud_io.anchor_left = 1
	hud_io.offset_left = -400
	hud_io.offset_top = 10
	hud_io.heading = "SENSORY IN (Hz)  →  DESCENDING OUT (Hz)"
	hud.add_child(hud_io)
	hud_ring = Ring.new()
	hud_ring.position = Vector2(14, 470)
	hud.add_child(hud_ring)
	hud_msg = Label.new()
	hud_msg.anchor_top = 1
	hud_msg.offset_top = -120
	hud_msg.position.x = 14
	hud_msg.add_theme_font_size_override("font_size", 13)
	hud_msg.add_theme_color_override("font_color", Color(0.8, 0.8, 0.8))
	hud_msg.text = "[L] loom a predator   [W] wind   [F] food   [B] bitter patch   [click] place food\n" \
		+ "[O] optogenetic P9/DNp09 walk drive   [T] move sun   [C] camera   [V] hide brain   [1/2/3] time 1x / 0.5x / 0.25x   [R] reset fly"
	hud.add_child(hud_msg)


# =================================================================== world queries (used by Fly)
func visual_objects() -> Array:
	var out := pillars.duplicate()
	if threat.visible:
		out.append({"pos": threat.position, "radius": 0.7, "height": threat.position.y + 0.7, "darkness": 1.0})
	return out


func odour_at(p: Vector3, kind: String) -> float:
	var c := 0.0
	for pt in patches:
		if pt["kind"] != kind:
			continue
		var d := Vector2(p.x - pt["pos"].x, p.z - pt["pos"].z)
		# plume skewed downwind
		var shift := Vector2(wind.x, wind.z) * 1.5
		d -= shift
		var sg := ODOUR_SIGMA if kind == "food" else ODOUR_SIGMA * 0.55
		c += pt["amount"] * exp(-d.length_squared() / (2.0 * sg * sg))
	return clampf(c, 0.0, 1.0)


var _grn_resp: Dictionary


## Chemistry under a point (taste organs): concentrations from data/grn_response.json "patches".
func chemistry_at(p: Vector3) -> Dictionary:
	if _grn_resp.is_empty():
		_grn_resp = JSON.parse_string(FileAccess.get_file_as_string("res://data/grn_response.json"))
	for pt in patches:
		if pt["amount"] > 0.0 and Vector2(p.x - pt["pos"].x, p.z - pt["pos"].z).length() < pt["radius"]:
			return _grn_resp["patches"].get(pt["kind"], {})
	return {}


func patch_under(p: Vector3) -> String:
	for pt in patches:
		if pt["amount"] > 0.05 and Vector2(p.x - pt["pos"].x, p.z - pt["pos"].z).length() < pt["radius"]:
			return pt["kind"]
	return ""


func consume(p: Vector3, dt: float) -> void:
	for pt in patches:
		if pt["kind"] == "food" and Vector2(p.x - pt["pos"].x, p.z - pt["pos"].z).length() < pt["radius"]:
			pt["amount"] = maxf(0.0, pt["amount"] - dt * 0.02)
			pt["node"].scale = Vector3.ONE * (0.3 + 0.7 * pt["amount"])


func threat_position() -> Vector3:
	return threat.position if threat.visible else fly.global_position + fly.global_transform.basis.z


func launch_threat() -> void:
	if fly == null:
		return
	var a := randf() * TAU
	if _args.has("threat_angle"):
		# degrees relative to the fly's heading: 0 = head-on, 90 = from its left, -90 = right
		a = -fly.rotation.y - PI / 2 - deg_to_rad(float(_args["threat_angle"]))
	threat.position = fly.global_position + Vector3(cos(a) * 9.0, 1.2, sin(a) * 9.0)
	threat_vel = (fly.global_position + Vector3(0, 0.3, 0) - threat.position).normalized() * 7.0
	threat.visible = true
	print("THREAT launched at t=%.2fs from %s" % [brain.sim_time_ms / 1000.0, _args.get("threat_angle", "random")])


# =================================================================== loop
func _process(dt: float) -> void:
	_frame += 1
	if error_text != "":
		_maybe_finish()
		return
	if threat.visible:
		threat.position += threat_vel * dt
		if threat.position.y < 0.2 or threat.position.length() > ARENA_R + 6:
			threat.visible = false

	fly.update_senses(dt)
	var fresh := brain.tick(dt)
	if fresh:
		fly.update_motor(dt)
		if act_tex:
			var img := Image.create_from_data(brain.meta["tex_width"], brain.n_pad / int(brain.meta["tex_width"]),
				false, Image.FORMAT_RF, brain.activity_bytes)
			act_tex.update(img)
	_update_camera(dt)
	_update_hud()
	_auto_script()
	if _args.has("record") and fresh:
		_record()
	if _args.has("probe") and fresh:
		_probe()
	if _args.has("eye_dump") and fly.eye != null and fresh and _frame % 5 == 0:
		_eye_dump()
	if _args.has("log"):
		_log_t += dt
		if _log_t >= 1.0:
			_log_t = 0.0
			_print_log()
	_maybe_finish()


func _update_camera(dt: float) -> void:
	var fp := fly.global_position
	var target: Vector3
	if cam_mode == 0:
		var back := fly.global_transform.basis.z
		var dist := float(_args.get("cam_dist", "2.4"))   # --cam_dist=0.6 for a close-up
		target = fp + back * dist + fly.global_transform.basis.x * dist * 0.5 + Vector3(0, dist * 0.55, 0)
		cam.position = cam.position.lerp(target, 1.0 - exp(-dt * 3.0))
		cam.look_at(fp + Vector3(0, 0.1, 0))
	else:
		cam.position = Vector3(0, 24, 0.01)
		cam.look_at(Vector3.ZERO)
	_orbit += dt * 0.25
	brain_cam.position = Vector3(sin(_orbit) * 7.5, 1.0, cos(_orbit) * 7.5)
	brain_cam.look_at(Vector3(0, -0.2, 0))


func _update_hud() -> void:
	var sim_ratio := brain.time_scale
	hud_stats.text = "connectome: %s  (%s)\nneurons %s   synaptic edges %s   model: LIF, dt %.1f ms\n" % [
		brain.meta["source"], brain.meta["note"].substr(0, 60) + "…", _thousands(brain.n), _thousands(brain.e), brain.meta["lif"]["dt_ms"]] \
		+ "emulated time %.1f s   speed %.2fx real-time   GPU batch %.1f ms   fps %d\n" % [
		brain.sim_time_ms / 1000.0, sim_ratio, brain.gpu_ms, Engine.get_frames_per_second()] \
		+ "whole-brain spikes/s %s   hunger %.2f   escapes %d   %s" % [
		_thousands(int(brain.total_spikes_per_s)), fly.hunger, fly.escapes,
		("OPTO P9 ON  " if fly.opto_p9 else "") + ("FEEDING" if fly.feeding else ("AIRBORNE" if fly.airborne > 0 else ("walking %.2f cm/s" % fly.speed)))]
	var rows := []
	for i in brain.n_regions:
		var r: Dictionary = brain.meta["regions"][i]
		rows.append([r["name"], brain.region_hz[i], 40.0, Color(r["color"])])
	hud_regions.rows = rows
	hud_regions.queue_redraw()
	var io := []
	var cin := Color(0.4, 0.8, 1.0)
	var cout := Color(1.0, 0.55, 0.3)
	io.append(["eye L (mean lum)", _mean_lum("L") * 100.0, 150.0, cin])
	io.append(["eye R (mean lum)", _mean_lum("R") * 100.0, 150.0, cin])
	io.append(["odour food L / R", fly.sense.get("odour_L", 0.0) * 100.0, 100.0, cin])
	io.append(["", fly.sense.get("odour_R", 0.0) * 100.0, 100.0, cin])
	io.append(["odour aversive L/R", fly.sense.get("bad_L", 0.0) * 100.0, 100.0, cin])
	io.append(["", fly.sense.get("bad_R", 0.0) * 100.0, 100.0, cin])
	io.append(["wind (JO) L / R", fly.sense.get("jo_L", 0.0), 300.0, cin])
	io.append(["", fly.sense.get("jo_R", 0.0), 300.0, cin])
	io.append(["loom (dL/dt)", fly.sense.get("loom", 0.0) * 10.0, 100.0, cin])
	io.append(["taste: " + str(fly.sense.get("taste", "")), 1.0 if fly.sense.get("taste", "") != "" else 0.0, 1.0, cin])
	io.append(["P9 forward L", brain.rate("dn_forward_L"), 80.0, cout])
	io.append(["P9 forward R", brain.rate("dn_forward_R"), 80.0, cout])
	io.append(["DNa02 turn L", brain.rate("dn_turn_L"), 80.0, cout])
	io.append(["DNa02 turn R", brain.rate("dn_turn_R"), 80.0, cout])
	io.append(["MDN backward", brain.rate("dn_backward"), 80.0, cout])
	io.append(["Giant fibre", brain.rate("dn_giant_fiber"), 80.0, cout])
	io.append(["MN9 proboscis", brain.rate("mn9_proboscis"), 80.0, cout])
	hud_io.rows = io
	hud_io.queue_redraw()
	var epg := PackedFloat32Array()
	for k in 8:
		epg.append(brain.rate("epg_%d" % k))
	hud_ring.epg = epg
	hud_ring.sun_rel = wrapf(sun_azimuth - fly.rotation.y, -PI, PI)
	hud_ring.queue_redraw()


func _mean_lum(side: String) -> float:
	var s := 0.0
	for k in 6:
		s += fly.sense.get("lum_%s%d" % [side, k], 0.0)
	return s / 6.0


func _unhandled_input(ev: InputEvent) -> void:
	if ev is InputEventKey and ev.pressed and not ev.echo:
		match ev.keycode:
			KEY_L: launch_threat()
			KEY_W:
				wind = Vector3.ZERO if wind.length() > 0 else Vector3(cos(randf() * TAU), 0, sin(randf() * TAU)) * 1.0
			KEY_F: add_patch(Vector3(randf_range(-8, 8), 0, randf_range(-8, 8)), "food")
			KEY_B: add_patch(Vector3(randf_range(-8, 8), 0, randf_range(-8, 8)), "aversive")
			KEY_T:
				sun_azimuth = wrapf(sun_azimuth + PI / 3, -PI, PI)
				_update_sun()
			KEY_C: cam_mode = 1 - cam_mode
			KEY_O: fly.opto_p9 = not fly.opto_p9
			KEY_V:
				if brain_view:
					brain_view.visible = not brain_view.visible
			KEY_1: brain.time_scale = 1.0
			KEY_2: brain.time_scale = 0.5
			KEY_3: brain.time_scale = 0.25
			KEY_R:
				fly.position = Vector3(-2, 0.06, 7)
				fly.hunger = 0.8
	elif ev is InputEventMouseButton and ev.pressed and ev.button_index == MOUSE_BUTTON_LEFT:
		var from := cam.project_ray_origin(ev.position)
		var dir := cam.project_ray_normal(ev.position)
		if absf(dir.y) > 1e-4:
			var hit := from + dir * (-from.y / dir.y)
			if Vector2(hit.x, hit.z).length() < ARENA_R - 0.6:
				add_patch(hit, "food")


# =================================================================== test harness
func _auto_script() -> void:
	if not _args.has("auto"):
		if _args.has("threat_at") and _frame == int(_args["threat_at"]):
			launch_threat()
		return
	# scripted events so a headless-ish run exercises every pathway
	if _frame == 2 and _args.has("opto"):
		fly.opto_p9 = true
	if _frame == 240:
		wind = Vector3(1, 0, 0)
	elif _frame == 420:
		wind = Vector3.ZERO
	if _frame == int(_args.get("threat_at", "600")):
		launch_threat()


func _print_log() -> void:
	var r := brain.region_hz
	if fly.legs != null and _args.has("legs_log"):
		var lf: Dictionary = fly.legs.angle["lf"]
		print("legs_dbg lf ThC %.2f CTr %.2f FTi %.2f TiTa %.2f | proprio %s" % [lf["ThC_pro"], lf["CTr"], lf["FTi"], lf["TiTa"], str(fly.legs.sense)])
	var s := "t=%.1fs spikes/s=%d gpu=%.1fms fps=%d | pos=(%.1f,%.1f) spd=%.2f turn=%.2f air=%s feed=%s esc=%d hunger=%.2f" % [
		brain.sim_time_ms / 1000.0, brain.total_spikes_per_s, brain.gpu_ms, Engine.get_frames_per_second(),
		fly.position.x, fly.position.z, fly.speed, fly.turn_rate, fly.airborne > 0, fly.feeding, fly.escapes, fly.hunger]
	s += "\n   DN: P9 %.1f/%.1f  DNa02 %.1f/%.1f  MDN %.1f  GF %.1f  MN9 %.1f | odourL/R %.2f/%.2f loom %.1f" % [
		brain.rate("dn_forward_L"), brain.rate("dn_forward_R"), brain.rate("dn_turn_L"), brain.rate("dn_turn_R"),
		brain.rate("dn_backward"), brain.rate("dn_giant_fiber"), brain.rate("mn9_proboscis"),
		fly.sense.get("odour_L", 0.0), fly.sense.get("odour_R", 0.0), fly.sense.get("loom", 0.0)]
	var parts := []
	for i in brain.n_regions:
		parts.append("%s=%.1f" % [brain.meta["regions"][i]["name"].get_slice(" ", 0) + brain.meta["regions"][i]["name"].right(1), r[i]])
	s += "\n   Hz: " + " ".join(parts)
	if fly.taste != null:
		s += "
   taste: %s | PER rostrum %.2f haustellum %.2f spread %.2f pump %.2f feeding %s" % [
			str(fly.taste.sense), fly.per_rostrum, fly.per_haustellum, fly.labellum_spread, fly.pumping, fly.feeding]
	if fly.eye != null:
		var el := fly.eye.summary("left")
		var er := fly.eye.summary("right")
		s += "
   eye: lumL/R %.3f/%.3f contrastL/R %+.2f/%+.2f R1-6 %.0f/%.0f Hz | " % [el.x, er.x, el.y, er.y, el.z, er.z]
		var vp := []
		for t in ["R1-6", "R7", "R8", "L1", "L2", "L3", "Mi1", "Tm3", "Tm1", "Tm2", "Tm9", "T4a", "T4b", "T4c", "T4d",
				"T5a", "T5b", "T5c", "T5d", "LC4", "LPLC1", "LPLC2", "LC6", "DNp01", "DNp02", "DNp11"]:
			vp.append("%s %.1f/%.1f" % [t, brain.pop_rate("type:%s_L" % t), brain.pop_rate("type:%s_R" % t)])
		s += "  ".join(vp)
	if _args.has("pops"):
		var pp := []
		for i in brain.n_pops:
			var nm: String = brain.meta["populations"][i]["name"]
			if not (nm.begins_with("PR_") or nm.begins_with("PRT_") or nm.begins_with("MED_") or nm.begins_with("OFF_") or nm.begins_with("CMP_") or nm.begins_with("EPG_")) or nm.ends_with("0") or nm.ends_with("3"):
				pp.append("%s=%.1f" % [nm, brain.pop_hz[i]])
		s += "\n   pops: " + " ".join(pp)
	print(s)


var _rec: FileAccess
var _rec_t := 0.0


func _record() -> void:
	## CSV: one row per 100 ms of emulated time with body state and every population's rate
	if _rec == null:
		var path := ProjectSettings.globalize_path("res://" + _args["record"])
		DirAccess.make_dir_recursive_absolute(path.get_base_dir())
		_rec = FileAccess.open(path, FileAccess.WRITE)
		var head := ["t_ms", "x", "z", "yaw", "speed", "turn", "airborne", "feeding", "hunger", "opto_p9", "proboscis", "per_rostrum", "per_haustellum", "labellum_spread", "pumping"]
		for pdef in brain.meta["populations"]:
			head.append(pdef["name"])
		_rec.store_csv_line(PackedStringArray(head))
	if brain.sim_time_ms - _rec_t < 100.0:
		return
	_rec_t = brain.sim_time_ms
	var row := PackedStringArray(["%.1f" % brain.sim_time_ms, "%.3f" % fly.position.x, "%.3f" % fly.position.z,
		"%.3f" % fly.rotation.y, "%.3f" % fly.speed, "%.3f" % fly.turn_rate, str(int(fly.airborne > 0)),
		str(int(fly.feeding)), "%.3f" % fly.hunger, str(int(fly.opto_p9)), "%.3f" % fly.proboscis, "%.3f" % fly.per_rostrum, "%.3f" % fly.per_haustellum,
		"%.3f" % fly.labellum_spread, "%.3f" % fly.pumping])
	for i in brain.n_pops:
		row.append("%.2f" % brain.pop_hz[i])
	_rec.store_csv_line(row)


var _eye_f: FileAccess
var _probe_f: FileAccess
var _probe_groups := {}


func _probe() -> void:
	## CSV: t_ms, threat visible, then per probe group the mean activity trace (spikes filtered
	## with a 60 ms exponential, per neuron) -> multiply by ~16.7 for an approximate rate in Hz
	if _probe_f == null:
		_probe_groups = JSON.parse_string(FileAccess.get_file_as_string("res://" + _args["probe"]))
		var path := ProjectSettings.globalize_path("res://" + _args.get("probe_out", "recordings/probe.csv"))
		_probe_f = FileAccess.open(path, FileAccess.WRITE)
		var head := PackedStringArray(["t_ms", "threat"])
		head.append_array(PackedStringArray(_probe_groups.keys()))
		_probe_f.store_csv_line(head)
	var act := brain.activity_bytes.to_float32_array()
	var row := PackedStringArray(["%.1f" % brain.sim_time_ms, str(int(threat.visible))])
	for k in _probe_groups:
		var ids: Array = _probe_groups[k]
		var s := 0.0
		for i in ids:
			s += act[int(i)]
		row.append("%.3f" % (s / maxf(ids.size(), 1) * 1000.0 / brain.act_tau_ms))
	_probe_f.store_csv_line(row)


func _eye_dump() -> void:
	## CSV per dump: t_ms, threat visible, then per facet luminance / contrast / R1-6 rate
	if _eye_f == null:
		var path := ProjectSettings.globalize_path("res://" + _args["eye_dump"])
		DirAccess.make_dir_recursive_absolute(path.get_base_dir())
		_eye_f = FileAccess.open(path, FileAccess.WRITE)
	var o := brain.eye_facet_out
	var row := PackedStringArray(["%.1f" % brain.sim_time_ms, str(int(threat.visible))])
	for i in o.size() / 4:
		row.append("%.4f;%.3f;%.1f" % [o[4 * i], o[4 * i + 1], o[4 * i + 2]])
	_eye_f.store_csv_line(row)


func _maybe_finish() -> void:
	if not _args.has("frames") or _frame < int(_args["frames"]):
		return
	if _args.has("shot"):
		var path := ProjectSettings.globalize_path("res://" + _args["shot"])
		DirAccess.make_dir_recursive_absolute(path.get_base_dir())
		get_viewport().get_texture().get_image().save_png(path)
		print("screenshot: ", path)
	if _rec:
		_rec.close()
	if brain.homeostasis_learning:
		brain.save_homeostasis()
		print("homeostasis: learned offsets saved to " + brain.homeo_path())
	if _eye_f:
		_eye_f.close()
	if _probe_f:
		_probe_f.close()
	brain.free_gpu()
	get_tree().quit()


func _exit_tree() -> void:
	brain.free_gpu()


static func _thousands(v: int) -> String:
	var s := str(v)
	var out := ""
	while s.length() > 3:
		out = "," + s.right(3) + out
		s = s.left(s.length() - 3)
	return s + out


# =================================================================== HUD widgets
class Bars extends Control:
	var rows: Array = []
	var heading := ""

	func _draw() -> void:
		var font := get_theme_default_font()
		var h := 18
		draw_rect(Rect2(-6, -4, 392, rows.size() * h + 30), Color(0, 0, 0, 0.45))
		draw_string(font, Vector2(0, 12), heading, HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.6, 0.8, 1.0))
		for i in rows.size():
			var r: Array = rows[i]
			var y := 20 + i * h
			draw_string(font, Vector2(0, y + 12), r[0], HORIZONTAL_ALIGNMENT_LEFT, 150, 12, Color(0.85, 0.85, 0.85))
			var frac := clampf(r[1] / r[2], 0.0, 1.0)
			draw_rect(Rect2(155, y + 3, 170, 11), Color(1, 1, 1, 0.07))
			draw_rect(Rect2(155, y + 3, 170 * frac, 11), r[3])
			draw_string(font, Vector2(332, y + 12), "%.1f" % r[1], HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.9, 0.9, 0.9))


class Ring extends Control:
	## Central-complex compass: EPG wedge activity vs. the true sun bearing.
	var epg := PackedFloat32Array()
	var sun_rel := 0.0

	func _draw() -> void:
		var font := get_theme_default_font()
		var c := Vector2(90, 100)
		draw_rect(Rect2(-6, -4, 392, 200), Color(0, 0, 0, 0.45))
		draw_string(font, Vector2(0, 12), "CENTRAL COMPLEX HEADING (EPG bump)", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.6, 0.8, 1.0))
		var mx := 1.0
		for v in epg:
			mx = maxf(mx, v)
		for k in epg.size():
			# wedge k encodes "sun at relative azimuth 2*pi*k/8" (+ = left, drawn counter-clockwise from up)
			var a := -PI / 2 - TAU * k / 8.0
			var v := epg[k] / mx
			var col := Color(0.95, 0.35, 0.3, 0.2 + 0.8 * v)
			draw_arc(c, 45 + 25 * v, a - 0.33, a + 0.33, 8, col, 10 + 14 * v)
		var sa := -PI / 2 - sun_rel
		draw_line(c, c + Vector2(cos(sa), sin(sa)) * 80, Color(1, 0.9, 0.3), 2)
		draw_circle(c, 4, Color.WHITE)
		draw_string(font, Vector2(190, 70), "yellow = true sun bearing", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(1, 0.9, 0.3))
		draw_string(font, Vector2(190, 90), "red = emulated EPG wedges", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.95, 0.45, 0.4))
		draw_string(font, Vector2(190, 110), "(up = fly's forward)", HORIZONTAL_ALIGNMENT_LEFT, -1, 12, Color(0.7, 0.7, 0.7))
