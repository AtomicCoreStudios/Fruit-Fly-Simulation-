#[compute]
#version 450
// Sensory receptor neurons driven by body-contact physics (taste, later touch/smell): the CPU
// computes a Poisson rate per receptor neuron each frame; this pass writes them into the brain's
// per-neuron input buffer (ext, shared with lif.glsl).
layout(local_size_x = 64) in;
layout(set = 0, binding = 0, std430) readonly buffer Idx  { int idx[]; };
layout(set = 0, binding = 1, std430) readonly buffer Rate { float rate[]; };
layout(set = 0, binding = 2, std430) buffer Ext            { float ext[]; };
layout(push_constant, std430) uniform Params { uint n; uint pad0; uint pad1; uint pad2; } p;
void main() {
	uint k = gl_GlobalInvocationID.x;
	if (k >= p.n) return;
	ext[uint(idx[k])] = rate[k];
}
