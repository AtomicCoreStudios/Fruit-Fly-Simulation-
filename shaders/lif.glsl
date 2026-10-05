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
// Electrical synapses (gap junctions): a presynaptic spike adds coupling (micro-volts) straight to
// the postsynaptic membrane voltage on the next step (mode 2 fills, mode 0 applies)
layout(set = 0, binding = 15, std430) buffer VKick         { int v_kick[]; };
layout(set = 0, binding = 16, std430) readonly buffer Gap  { ivec4 gap[]; };   // pre, post, micro-volts, 0
// Intrinsic physiology (tools/build_physiology.py, data/physiology_rules.json): per neuron
// (mu = resting offset mV, sigma = OU membrane noise mV, b = adaptation increment mV, tau_w ms)
layout(set = 0, binding = 17, std430) readonly buffer Phys { vec4 phys[]; };
layout(set = 0, binding = 18, std430) buffer State2        { vec2 st2[]; };     // (noise x, adaptation w)
const float TAU_NOISE = 5.0;   // ms, data/physiology_rules.json "noise.tau_ms"
// Homeostatic intrinsic plasticity: (target rate Hz, learned offset mV). While learning (mode 0 with
// p.pad == 1), the offset moves the neuron's excitability toward its set point; afterwards it is frozen.
layout(set = 0, binding = 19, std430) buffer Homeo         { vec2 homeo[]; };
const float ETA_H = 0.0001;    // mV per ms per Hz of rate error
// Range of the learned offset. Real intrinsic plasticity is a few mV, but a +-5 mV cap lets the antennal lobe
// run away in this point-neuron model (README "Leg sugar"), so the wide range is kept: basis approximate.
const float HOMEO_MIN = -40.0;
const float HOMEO_MAX = 15.0;
// Presynaptic release gain (neuromodulation of transmitter release, e.g. dopamine via DopEcR on sugar
// GRN terminals in hungry flies, Inagaki et al. 2012): multiplies every outgoing synaptic weight; 1 = none.
layout(set = 0, binding = 20, std430) readonly buffer Release { float release_gain[]; };
// Conductance-based synapses (p.cond == 1): inhibitory input gets its own ring buffer and state. Weights stay in
// mV units (Shiu et al.'s 0.275 mV per synapse); they are converted to leak-normalised conductances
// a = g_e / (E_exc - v_n), b = g_i / (v_n - E_inh) with v_n = (v_rest + v_th)/2, so a synapse has exactly
// Shiu's effect in the middle of the subthreshold range (where his weight was fitted to behaviour) and
// differs away from it (excitation saturates near E_exc, inhibition shunts and reverses at E_inh). Checked
// against tools/reference_lif.py scheme "cond": Shiu sugar set 100 Hz -> MN9 69.5/52 Hz (current: 68/53).
layout(set = 0, binding = 21, std430) buffer GInh  { float g_inh[]; };
layout(set = 0, binding = 22, std430) buffer GInI  { int g_in_i[]; };
// Non-spiking (graded) neurons, per neuron vec4 (r_max Hz, gain Hz/mV, threshold mV above rest, 0); r_max 0 =
// ordinary spiking neuron. Graded neurons never spike, reset or go refractory.
//  gain == 0: sigmoid release f(v) = r_max / (1 + exp(-(v - V_HALF)/K)), stochastic events (APL, Papadopoulou et
//             al. 2011; patchy AL LNs lLN2P, Schenk & Gaudry 2023)
//  gain  > 0: rate unit of Pugliese et al. 2025 (VNC CPG model): f = max(0, r_max tanh(gain (v - v_rest - thr)
//             / r_max)), released deterministically (accumulator in st2.y), no membrane noise
layout(set = 0, binding = 23, std430) readonly buffer Graded { vec4 graded[]; };
const float RATE_Q = 20.0;     // rate units release in quanta of weight/RATE_Q (smooth, rate-like transmission)
const float GRADED_V_HALF = -45.0;
const float GRADED_K = 2.0;

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
	uint pad;          // mode 2: number of gap junctions
	float e_exc;       // mV, excitatory reversal (cond == 1)
	float e_inh;       // mV, inhibitory reversal
	uint cond;         // 1: conductance-based synapses; 0: Shiu et al. current-like synapses
} p;

uint pcg(uint x) {
	uint s = x * 747796405u + 2891336453u;
	uint w2 = ((s >> ((s >> 28u) + 4u)) ^ s) * 277803737u;
	return (w2 >> 22u) ^ w2;
}

float gauss(uint i, uint step) {
	uint a = pcg(i * 2654435761u ^ pcg(step * 2246822519u + 7u));
	uint c = pcg(a + 0x9E3779B9u);
	float u1 = (float(a & 0xFFFFFFu) + 1.0) / 16777217.0;
	float u2 = float(c & 0xFFFFFFu) / 16777216.0;
	return sqrt(-2.0 * log(u1)) * cos(6.2831853 * u2);
}

void main() {
	uint i = gl_GlobalInvocationID.x;
	if (i >= p.n) return;

	if (p.mode == 0u) {
		// Shiu et al. 2024 semantics (data/raw/model.py, Brian2), checked against
		// tools/reference_lif.py: exact ('linear') integration of
		//   dv/dt = (v_0 - v + g + bias)/t_mbr,  dg/dt = -g/tau   (both frozen while refractory)
		// Poisson input kicks v directly (PoissonInput target_var='v'), stimulated neurons have no
		// refractory period, and a spike resets v and g.
		uint slot = (p.step % p.slots) * p.n;
		float gi = g[i] + float(atomicExchange(g_in[slot + i], 0)) * 0.001;
		int gid = group_id[i];
		float rate = group_rate[gid] + ext[i];
		vec4 ph = phys[i];
		vec2 s2 = st2[i];
		if (ph.y > 0.0) {
			float en = exp(-p.dt / TAU_NOISE);
			s2.x = s2.x * en + ph.y * sqrt(1.0 - en * en) * gauss(i, p.step);
		}
		bool rate_unit = graded[i].y > 0.0;           // st2.y is then the release accumulator
		if (!rate_unit) s2.y *= exp(-p.dt / max(ph.w, 1.0));
		vec2 hm = homeo[i];
		if (p.pad == 1u && hm.x >= 0.0) {
			float r_est = activity[i] / 0.06;   // spikes filtered with 60 ms time constant -> Hz
			hm.y = clamp(hm.y + ETA_H * p.dt * (hm.x - r_est), HOMEO_MIN, HOMEO_MAX);
			homeo[i] = hm;
		}
		float bias = ext[p.n + i] + ph.x + s2.x - (rate_unit ? 0.0 : s2.y) + hm.y;
		float vi = v[i] + float(atomicExchange(v_kick[i], 0)) * 0.001;
		bool not_refr = refrac[i] <= 0.0;
		float gh = 0.0;
		if (p.cond == 1u) gh = g_inh[i] + float(atomicExchange(g_in_i[slot + i], 0)) * 0.001;
		if (not_refr) {
			float es = p.syn_decay;
			if (p.cond == 1u && rate_unit) {
				// rate units (Pugliese et al. model): linear current-based input, excitation minus inhibition
				float em = exp(-p.dt / p.tau_m);
				float tau_s = -p.dt / log(es);
				float k = tau_s / (p.tau_m - tau_s);
				float u = vi - p.v_rest - bias;
				vi = p.v_rest + bias + u * em + (gi - gh) * k * (em - es);
				gh *= es;
			} else if (p.cond == 1u) {
				// conductances at the step midpoint, exact exponential relaxation with total conductance
				float sh = sqrt(es);
				float vn = 0.5 * (p.v_rest + p.v_th);
				float a = max(gi, 0.0) * sh / (p.e_exc - vn);
				float b = max(gh, 0.0) * sh / (vn - p.e_inh);
				float gt = 1.0 + a + b;
				float vinf = (p.v_rest + bias + a * p.e_exc + b * p.e_inh) / gt;
				vi = vinf + (vi - vinf) * exp(-p.dt * gt / p.tau_m);
				gh *= es;
			} else {
				float em = exp(-p.dt / p.tau_m);
				float tau_s = -p.dt / log(es);
				float k = tau_s / (p.tau_m - tau_s);
				float u = vi - p.v_rest - bias;
				vi = p.v_rest + bias + u * em + gi * k * (em - es);
			}
			gi *= es;
		}
		if (rate > 0.0) {
			float r = float(pcg(i * 9781u ^ pcg(p.step)) & 0xFFFFFFu) / 16777216.0;
			if (r < rate * p.dt * 0.001) vi += p.w_poisson;
		}
		refrac[i] -= p.dt;
		uint s = 0u;
		vec4 gp = graded[i];
		if (gp.x > 0.0) {
			// graded release event (counted as a "spike" for propagation and readout; no reset)
			if (gp.y > 0.0) {
				float f = max(0.0, gp.x * tanh(gp.y * (vi - p.v_rest - gp.z) / gp.x));
				s2.y += f * p.dt * 0.001 * RATE_Q;
				float nq = floor(s2.y);
				if (nq >= 1.0) {
					s2.y -= nq;
					s = uint(min(nq, 255.0));   // number of quanta this step
				}
			} else {
				float f = gp.x / (1.0 + exp(-(vi - GRADED_V_HALF) / GRADED_K));
				float r = float(pcg(i * 7919u ^ pcg(p.step + 104729u)) & 0xFFFFFFu) / 16777216.0;
				if (r < f * p.dt * 0.001) s = 1u;
			}
			if (s >= 1u) {
				atomicAdd(group_count[gid], 1u);
				atomicAdd(pop_count[pop_id[i]], 1u);
			}
			refrac[i] = 0.0;
		} else if (not_refr && vi > p.v_th) {
			s = 1u;
			vi = p.v_reset;
			gi = 0.0;
			gh = 0.0;
			refrac[i] = rate > 0.0 ? 0.0 : p.t_ref;   // Poisson-stimulated neurons: no refractory period
			s2.y += ph.z;                              // spike-frequency adaptation
			atomicAdd(group_count[gid], 1u);
			atomicAdd(pop_count[pop_id[i]], 1u);
		}
		v[i] = vi;
		g[i] = gi;
		if (p.cond == 1u) g_inh[i] = gh;
		st2[i] = s2;
		spiked[i] = s;
		activity[i] = activity[i] * p.act_decay + ((gp.y > 0.0) ? float(s) / RATE_Q : float(s));
	} else if (p.mode == 2u) {
		if (i >= p.pad) return;                 // pad = number of gap junctions
		ivec4 gj = gap[i];
		if (spiked[uint(gj.x)] != 0u) atomicAdd(v_kick[uint(gj.y)], gj.z);
	} else {
		if (spiked[i] == 0u) return;
		uint slot = ((p.step + p.delay) % p.slots) * p.n;
		int e1 = row_ptr[i + 1u];
		float rg = release_gain[i];
		if (graded[i].y > 0.0) rg *= float(spiked[i]) / RATE_Q;   // rate unit: quanta of weight/RATE_Q
		if (p.cond == 1u) {
			for (int e = row_ptr[i]; e < e1; e++) {
				int wv = (rg == 1.0) ? w[e] : int(float(w[e]) * rg);
				if (wv >= 0) atomicAdd(g_in[slot + uint(col[e])], wv);
				else atomicAdd(g_in_i[slot + uint(col[e])], -wv);
			}
		} else if (rg == 1.0) {
			for (int e = row_ptr[i]; e < e1; e++) {
				atomicAdd(g_in[slot + uint(col[e])], w[e]);
			}
		} else {
			for (int e = row_ptr[i]; e < e1; e++) {
				atomicAdd(g_in[slot + uint(col[e])], int(float(w[e]) * rg));
			}
		}
	}
}
