#[compute]
#version 450
// Whole-brain leaky integrate-and-fire step (Shiu et al. 2024 style).
// mode 0: integrate membranes, emit spikes, inject Poisson sensory drive
// mode 1: propagate spikes along CSR synapse lists (fixed-point atomics)

layout(local_size_x = 256) in;

layout(set = 0, binding = 0, std430) buffer V        { float v[]; };
layout(set = 0, binding = 1, std430) buffer G        { float g[]; };
layout(set = 0, binding = 2, std430) buffer Refrac   { float refrac[]; };
layout(set = 0, binding = 3, std430) buffer GIn      { int g_in[]; };
layout(set = 0, binding = 4, std430) buffer Spiked   { uint spiked[]; };
layout(set = 0, binding = 5, std430) readonly buffer Group   { int group_id[]; };
layout(set = 0, binding = 6, std430) readonly buffer Pop     { int pop_id[]; };
layout(set = 0, binding = 7, std430) readonly buffer Rates   { float group_rate[]; };
layout(set = 0, binding = 8, std430) readonly buffer RowPtr  { int row_ptr[]; };
layout(set = 0, binding = 9, std430) readonly buffer Col     { int col[]; };
layout(set = 0, binding = 10, std430) readonly buffer W      { int w[]; };
layout(set = 0, binding = 11, std430) buffer Activity { float activity[]; };
layout(set = 0, binding = 12, std430) buffer GCount   { uint group_count[]; };
layout(set = 0, binding = 13, std430) buffer PCount   { uint pop_count[]; };
// Per-neuron external input: ext[i] = Poisson rate (Hz), ext[n + i] = bias current (mV).
// Written by the eye pass (photoreceptor rates) and at start-up (lamina resting bias).
layout(set = 0, binding = 14, std430) readonly buffer Ext { float ext[]; };

layout(push_constant, std430) uniform Params {
	uint n;
	uint mode;
	uint step;
	float dt;          // ms
	float tau_m;       // ms
	float syn_decay;   // exp(-dt/tau_syn)
	float v_rest;
	float v_reset;
	float v_th;
	float t_ref;       // ms
	float w_poisson;   // mV per Poisson event
	float act_decay;   // viewer trace decay per step
	uint delay;        // synaptic delay in steps (>= 1)
	uint slots;        // ring-buffer slots = delay + 1
	uint reset_g;      // 1: clear synaptic input on spike (Shiu et al.)
	uint pad;
} p;

uint pcg(uint x) {
	uint s = x * 747796405u + 2891336453u;
	uint w2 = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
	return (w2 >> 22u) ^ w2;
}

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= p.n) return;

	if (p.mode == 0u) {
		uint slot = (p.step % p.slots) * p.n;
		float gin = float(atomicExchange(g_in[slot + i], 0)) * 0.001;
		int gid = group_id[i];
		float rate = group_rate[gid] + ext[i];
		if (rate > 0.0) {
			float u = float(pcg(i * 9781u ^ pcg(p.step)) & 0xFFFFFFu) / 16777216.0;
			if (u < rate * p.dt * 0.001) gin += p.w_poisson;
		}
		float gi = g[i] * p.syn_decay + gin;
		g[i] = gi;
		uint s = 0u;
		float vi = v[i];
		if (refrac[i] > 0.0) {
			refrac[i] -= p.dt;
			vi = p.v_reset;
		} else {
			vi += p.dt / p.tau_m * (p.v_rest - vi + gi + ext[p.n + i]);
			if (vi >= p.v_th) {
				s = 1u;
				vi = p.v_reset;
				refrac[i] = p.t_ref;
				if (p.reset_g == 1u) g[i] = 0.0;
				atomicAdd(group_count[gid], 1u);
				atomicAdd(pop_count[pop_id[i]], 1u);
			}
		}
		v[i] = vi;
		spiked[i] = s;
		activity[i] = activity[i] * p.act_decay + float(s);
	} else {
		if (spiked[i] == 0u) return;
		uint slot = ((p.step + p.delay) % p.slots) * p.n;
		int e1 = row_ptr[i + 1u];
		for (int e = row_ptr[i]; e < e1; e++) {
			atomicAdd(g_in[slot + uint(col[e])], w[e]);
		}
	}
}
