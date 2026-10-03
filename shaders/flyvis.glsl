#[compute]
#version 450
// Graded optic lobe: FlyVis (Lappalainen et al. 2024) pretrained network, two instances (right eye,
// left eye) of 45,669 cells each. Exact Euler step of PPNeuronIGRSynapses:
//   V += dt / max(tau, dt) * (-V + bias + sum_j w_ij relu(V_j) + x)
// x = facet luminance on the R1-R8 cells of each column (verified vs FlyVis to 1e-6 in numpy).
// mode 0: integrate one step (ping-pong state, parity selects read half)
// mode 1: read out cells shared with the FlyWire column map -> Poisson rates of the real neurons

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) buffer State          { float V[]; };      // 2 halves x 2 eyes x N
layout(set = 0, binding = 1, std430) readonly buffer Bias  { float bias[]; };
layout(set = 0, binding = 2, std430) readonly buffer Tau   { float tau[]; };
// FlyVis is convolutional: cells of a type share one filter of (source type, du, dv, weight bits).
layout(set = 0, binding = 3, std430) readonly buffer Cell  { ivec4 cell[]; };     // type, u, v, 0
layout(set = 0, binding = 4, std430) readonly buffer FPtr  { int filt_ptr[]; };
layout(set = 0, binding = 5, std430) readonly buffer Filt  { ivec4 filt[]; };
layout(set = 0, binding = 11, std430) readonly buffer Grid { int grid[]; };      // (type,u,v) -> cell
layout(set = 0, binding = 6, std430) readonly buffer InF   { int in_facet[]; }; // 2N, -1 = no input
layout(set = 0, binding = 7, std430) readonly buffer Facet { vec4 facet_out[]; }; // from eye.glsl
layout(set = 0, binding = 8, std430) readonly buffer Pairs { ivec2 out_pairs[]; }; // (FlyWire idx, node)
layout(set = 0, binding = 9, std430) readonly buffer Base  { float out_base[]; };
layout(set = 0, binding = 10, std430) buffer Ext           { float ext[]; };     // shared with lif.glsl

layout(push_constant, std430) uniform Params {
	uint n;        // cells per eye
	uint mode;
	uint parity;   // 0: read half 0, write half 1
	uint n_out;
	float dt;      // s
	float gain;    // Hz per unit of activity above grey
	float r_max;
	float pad;
} p;

void main() {
	uint i = gl_GlobalInvocationID.x;
	uint n2 = 2u * p.n;
	uint rd = p.parity * n2;
	if (p.mode == 0u) {
		if (i >= n2) return;
		uint eye = i / p.n;
		uint j = i - eye * p.n;
		ivec4 c = cell[j];
		float syn = 0.0;
		uint base = rd + eye * p.n;
		for (int k = filt_ptr[c.x]; k < filt_ptr[c.x + 1]; k++) {
			ivec4 fk = filt[k];
			int su = c.y - fk.y;
			int sv = c.z - fk.z;
			if (abs(su) > 15 || abs(sv) > 15 || abs(su + sv) > 15) continue;
			int s = grid[fk.x * 961 + (su + 15) * 31 + (sv + 15)];
			if (s >= 0) syn += intBitsToFloat(fk.w) * max(V[base + uint(s)], 0.0);
		}
		float x = 0.0;
		int f = in_facet[i];
		if (f >= 0) x = pow(max(facet_out[f].x, 0.0), 1.0 / 2.2);   // display-referred, like FlyVis' movies
		float v = V[rd + i];
		v += p.dt / max(tau[j], p.dt) * (-v + bias[j] + syn + x);
		V[(1u - p.parity) * n2 + i] = v;
	} else {
		if (i >= p.n_out) return;
		ivec2 pr = out_pairs[i];
		float a = max(V[rd + uint(pr.y)], 0.0) - out_base[i];
		ext[pr.x] = clamp(p.gain * a, 0.0, p.r_max);
	}
}
