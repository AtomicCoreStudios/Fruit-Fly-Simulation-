shader_type spatial;
render_mode unshaded, blend_add, depth_draw_never, cull_disabled, shadows_disabled, fog_disabled;

// One billboarded glow per neuron. Colour = neuropil, brightness/size = recent spiking.
uniform sampler2D activity : filter_nearest;
uniform int tex_width = 1024;
uniform float base_size = 0.013;
uniform float gain = 0.35;
uniform float idle = 0.22;

varying vec3 tint;
varying float act;

void vertex() {
	int id = INSTANCE_ID;
	float a = texelFetch(activity, ivec2(id % tex_width, id / tex_width), 0).r;
	act = 1.0 - exp(-a * gain);
	MODELVIEW_MATRIX = VIEW_MATRIX * mat4(INV_VIEW_MATRIX[0], INV_VIEW_MATRIX[1], INV_VIEW_MATRIX[2], MODEL_MATRIX[3]);
	VERTEX *= base_size * (1.0 + 2.5 * act);
	tint = COLOR.rgb;
}

void fragment() {
	float d = length(UV - 0.5) * 2.0;
	float m = smoothstep(1.0, 0.0, d);
	vec3 c = mix(tint * idle, mix(tint, vec3(1.0, 0.96, 0.85), 0.35) * 0.9, act) * 0.55;
	ALBEDO = c * m * m;
}
