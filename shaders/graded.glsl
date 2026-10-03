#[compute]
#version 450
// Graded optic lobe beyond FlyVis + graded VPN dendrites (tools/graded/build_graded.py).
// Same equations as tools/graded/validate.py:
//   tau dV/dt = -V + b + sum_j w_ij relu(A_j),  A = [FlyVis-driven cells | graded units]
// mode 0: gather the FlyVis activity of the driven FlyWire cells into A[0..nD)
// mode 1: Euler step of every graded unit (reads A, writes V)
// mode 2: A[nD+i] = relu(V_i); outputs to the spiking brain:
//         non-VPN graded cell -> Poisson rate of its FlyWire copy (gain * relu(V))
//         VPN dendrite        -> bias current (mV) c * V into its spiking FlyWire soma

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) buffer Act             { float A[]; };
layout(set = 0, binding = 1, std430) buffer Volt            { float Vg[]; };
layout(set = 0, binding = 2, std430) readonly buffer Tau    { float tau[]; };
layout(set = 0, binding = 3, std430) readonly buffer Bias   { float bias[]; };
layout(set = 0, binding = 4, std430) readonly buffer Row    { int row_ptr[]; };
layout(set = 0, binding = 5, std430) readonly buffer Col    { int col[]; };
layout(set = 0, binding = 6, std430) readonly buffer Wt     { float w[]; };
layout(set = 0, binding = 7, std430) readonly buffer DNode  { int d_node[]; };
layout(set = 0, binding = 8, std430) readonly buffer FvV    { float fvV[]; };   // FlyVis state (flyvis.glsl)
layout(set = 0, binding = 9, std430) readonly buffer GFw    { int g_fw[]; };
layout(set = 0, binding = 10, std430) readonly buffer GVpn  { int g_vpn[]; };
layout(set = 0, binding = 11, std430) buffer Ext            { float ext[]; };   // shared with lif.glsl

shared float red[256];

layout(push_constant, std430) uniform Params {
	uint n_d;
	uint n_g;
	uint mode;
	uint fv_off;   // offset of the freshest FlyVis half in fvV
	float dt;      // s
	float gain;    // Hz per unit (non-VPN graded cells -> Poisson)
	float c_mv;    // mV per unit (VPN dendrite -> soma bias)
	uint n_lif;    // neurons in the LIF brain (bias lives at ext[n_lif + i])
} p;

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (p.mode == 0u) {
		if (i >= p.n_d) return;
		A[i] = max(fvV[p.fv_off + uint(d_node[i])], 0.0);
	} else if (p.mode == 1u) {
		// 32 threads per graded unit (8 units per 256-thread workgroup): lanes stride through the
		// unit's synapse list (coalesced reads), then a shared-memory tree reduction
		uint lane = gl_LocalInvocationID.x & 31u;
		uint slot = gl_LocalInvocationID.x >> 5u;
		uint r = gl_WorkGroupID.x * 8u + slot;
		float s = 0.0;
		if (r < p.n_g)
			for (int e = row_ptr[r] + int(lane); e < row_ptr[r + 1u]; e += 32)
				s += w[e] * A[col[e]];
		red[gl_LocalInvocationID.x] = s;
		barrier();
		for (uint k = 16u; k > 0u; k >>= 1u) {
			if (lane < k) red[gl_LocalInvocationID.x] += red[gl_LocalInvocationID.x + k];
			barrier();
		}
		if (lane == 0u && r < p.n_g) {
			float v = Vg[r];
			v += p.dt / max(tau[r], p.dt) * (-v + bias[r] + red[gl_LocalInvocationID.x]);
			Vg[r] = v;
		}
	} else {
		if (i >= p.n_g) return;
		float v = Vg[i];
		A[p.n_d + i] = max(v, 0.0);
		uint fw = uint(g_fw[i]);
		if (g_vpn[i] == 1) {
			ext[fw] = 0.0;
			ext[p.n_lif + fw] = p.c_mv * v;
		} else {
			ext[fw] = clamp(p.gain * max(v, 0.0), 0.0, 250.0);
		}
	}
}
