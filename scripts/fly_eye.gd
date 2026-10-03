class_name FlyEye
extends Node
## The compound eyes: six 90-degree cameras form a cube map centred between the eyes and
## locked to the head. Every frame the cube is read back and handed to the brain, whose
## eye pass (shaders/eye.glsl) samples each of the 1,709 measured ommatidia and drives the
## real FlyWire photoreceptors of that facet's column. Data: data/eye_drive.json.

const BODY_LAYER := 2          # render layer of the fly's own body (hidden from the eyes)

var brain: FlyBrain
var head: Node3D
var drive: Dictionary
var px := 48
var _views: Array[SubViewport] = []
var _cams: Array[Camera3D] = []
var _faces: Array = []
var _centre := Vector3.ZERO
var _ready_frames := 0


func setup(p_brain: FlyBrain, p_head: Node3D, lamina_bias_mv: float, columnar_bias_mv := 0.0) -> String:
	brain = p_brain
	head = p_head
	var f := FileAccess.open("res://data/eye_drive.json", FileAccess.READ)
	if f == null:
		return "missing data/eye_drive.json (run tools/build_eye_drive.py)"
	drive = JSON.parse_string(f.get_as_text())
	px = int(drive["cube_px"])
	var c: Array = drive["cube_centre_head"]
	_centre = Vector3(c[0], c[1], c[2])
	_faces = drive["faces"]
	for i in 6:
		var vp := SubViewport.new()
		vp.size = Vector2i(px, px)
		vp.render_target_update_mode = SubViewport.UPDATE_ALWAYS
		vp.msaa_3d = Viewport.MSAA_DISABLED
		vp.positional_shadow_atlas_size = 0
		add_child(vp)
		var cam := Camera3D.new()
		cam.fov = 90.0
		cam.near = 0.003          # cm (30 um)
		cam.far = 200.0
		cam.cull_mask = 0xFFFFF & ~BODY_LAYER
		vp.add_child(cam)
		cam.current = true
		_views.append(vp)
		_cams.append(cam)
	var err := brain.init_eye(drive)
	if err != "":
		return err
	var lam: Array = []
	for t in drive["lamina"]:
		lam.append_array(drive["lamina"][t])
	# Lamina neurons are graded, non-spiking cells held depolarised at rest; in this LIF model a
	# steady bias stands in for that resting depolarisation so that histaminergic photoreceptor
	# inhibition can modulate them. basis: approximate (value tuned, see README).
	brain.set_neuron_bias(lam, lamina_bias_mv)
	# Same stand-in for the medulla/lobula-plate columnar types (Mi, Tm, C, T1-T5): a resting
	# depolarisation below threshold (7 mV) so their graded synaptic input can reach spiking.
	# basis: approximate - free parameter, swept in the loom test (README).
	if columnar_bias_mv != 0.0:
		var colm: Array = []
		for t in drive["columnar"]:
			colm.append_array(drive["columnar"][t])
		brain.set_neuron_bias(colm, columnar_bias_mv)
	return ""


func _face_basis(i: int) -> Basis:
	var fw: Array = _faces[i][0]
	var rt: Array = _faces[i][1]
	var up: Array = _faces[i][2]
	var x := Vector3(rt[0], rt[1], rt[2])
	var y := Vector3(up[0], up[1], up[2])
	var z := -Vector3(fw[0], fw[1], fw[2])   # cameras look down -Z
	return Basis(x, y, z)


## Call once per frame, before brain.tick().
func update(dt: float) -> void:
	var hx := head.global_transform
	var origin := hx * _centre
	var hb := hx.basis.orthonormalized()
	for i in 6:
		_cams[i].global_transform = Transform3D(hb * _face_basis(i), origin)
	_ready_frames += 1
	if _ready_frames < 3:
		return            # viewports have not rendered yet
	var bytes := PackedByteArray()
	for vp in _views:
		var img := vp.get_texture().get_image()
		if img.get_format() != Image.FORMAT_RGBA8:
			img.convert(Image.FORMAT_RGBA8)
		bytes.append_array(img.get_data())
	brain.set_eye_pixels(bytes, dt)


## Mean luminance, contrast and R1-6 rate over the facets of one eye (for the HUD/log).
func summary(side: String) -> Vector3:
	var out := brain.eye_facet_out
	if out.size() == 0:
		return Vector3.ZERO
	var s := Vector3.ZERO
	var k := 0
	var sides: Array = drive["facet_side"]
	for i in sides.size():
		if sides[i] == side:
			s += Vector3(out[4 * i], out[4 * i + 1], out[4 * i + 2])
			k += 1
	return s / maxf(k, 1)
