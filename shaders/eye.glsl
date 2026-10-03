#[compute]
#version 450
// Compound-eye pass: one thread per ommatidium (facet).
// 1. Sample the facet's Gaussian acceptance cone from the 6-face cube render (precomputed
//    pixel taps + weights, tools/build_eye_drive.py).
// 2. Photoreceptor adaptation: each channel adapts to its own recent mean (Weber-like).
// 3. Write a Poisson rate for every real FlyWire photoreceptor (R1-6, R7, R8) in this facet's
//    column into the brain's per-neuron input buffer.
// Approximation: real photoreceptors are graded, not spiking; here their output is a rate code.

layout(local_size_x = 64) in;

layout(set = 0, binding = 0, std430) readonly buffer Pix   { uint pix[]; };     // RGBA8 per pixel
layout(set = 0, binding = 1, std430) readonly buffer KPtr  { int k_ptr[]; };
layout(set = 0, binding = 2, std430) readonly buffer KPix  { int k_pix[]; };
layout(set = 0, binding = 3, std430) readonly buffer KW    { float k_w[]; };
layout(set = 0, binding = 4, std430) buffer Adapt          { vec4 adapt[]; };   // per facet: lum, UV*, blue, green
layout(set = 0, binding = 5, std430) readonly buffer PPtr  { int pr_ptr[]; };
layout(set = 0, binding = 6, std430) readonly buffer PNeu  { int pr_neuron[]; };
layout(set = 0, binding = 7, std430) readonly buffer PChan { int pr_chan[]; };
layout(set = 0, binding = 8, std430) buffer Ext            { float ext[]; };    // shared with lif.glsl
layout(set = 0, binding = 9, std430) buffer Out            { vec4 facet_out[]; };

layout(push_constant, std430) uniform Params {
	uint n_facets;
	uint first_frame;
	float dt;        // s since last eye update
	float tau_adapt; // s
	float r0;        // Hz at the adapted level
	float gain;      // rate change per unit Weber contrast
	float r_max;     // Hz
	float pad;
} p;

vec3 srgb_to_linear(vec3 c) { return pow(c, vec3(2.2)); }

void main() {
	uint f = gl_GlobalInvocationID.x;
	if (f >= p.n_facets) return;
	vec3 rgb = vec3(0.0);
	for (int k = k_ptr[f]; k < k_ptr[f + 1]; k++) {
		uint c = pix[k_pix[k]];
		vec3 px = vec3(float(c & 255u), float((c >> 8u) & 255u), float((c >> 16u) & 255u)) / 255.0;
		rgb += k_w[k] * srgb_to_linear(px);
	}
	// channels: 0 broadband (Rh1-like), 1 UV stand-in, 2 blue, 3 green
	vec4 I = vec4(dot(rgb, vec3(0.25, 0.55, 0.20)), rgb.b, rgb.b, rgb.g);
	vec4 A = p.first_frame == 1u ? I : adapt[f];
	A += (I - A) * (1.0 - exp(-p.dt / p.tau_adapt));
	adapt[f] = A;
	vec4 contrast = (I - A) / (A + 0.02);
	vec4 rate = clamp(p.r0 * (1.0 + p.gain * contrast), vec4(0.0), vec4(p.r_max));
	facet_out[f] = vec4(I.x, contrast.x, rate.x, 0.0);
	for (int j = pr_ptr[f]; j < pr_ptr[f + 1]; j++) {
		ext[pr_neuron[j]] = rate[pr_chan[j]];
	}
}
