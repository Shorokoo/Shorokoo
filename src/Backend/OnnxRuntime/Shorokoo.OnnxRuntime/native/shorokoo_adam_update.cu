// The CUDA kernel of the fused AdamUpdate (shorokoo_ort_ops.cpp): one thread per element, each
// element's operations in float32 and in the order of the chain of ONNX operators it stands for,
//
//   m' = beta1 * m + c1 * g
//   v' = beta2 * v + (c2 * g) * g
//   p' = p [* decay] - m' / (sqrt(v') + eps) * step
//
// each rounded to nearest as the operator it stands for rounds it: the intrinsics below are never
// contracted into a fused multiply-add, and the division and square root are the correctly
// rounded ones ONNX Runtime's CUDA Div and Sqrt compute. Built with --fmad=false besides (see
// CMakeLists.txt), so that nothing written later contracts either.
//
// An output may be the memory of the input it replaces: a thread reads its element of every input
// before it writes that element of any output, and no thread touches another's element.

#include <cuda_runtime.h>

#include "shorokoo_adam_update.h"

namespace {

constexpr unsigned Threads = 256;

template <bool Decays>
__global__ void AdamUpdate(ShorokooAdamUpdate u) {
    const size_t i = static_cast<size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= u.count) return;
    const float g = u.g[i];
    const float m = __fadd_rn(__fmul_rn(*u.beta1, u.m[i]), __fmul_rn(*u.c1, g));
    const float v = __fadd_rn(__fmul_rn(*u.beta2, u.v[i]), __fmul_rn(__fmul_rn(*u.c2, g), g));
    const float p = Decays ? __fmul_rn(u.p[i], *u.decay) : u.p[i];
    const float step = __fmul_rn(__fdiv_rn(m, __fadd_rn(__fsqrt_rn(v), *u.eps)), *u.step);
    u.mOut[i] = m;
    u.vOut[i] = v;
    u.pOut[i] = __fsub_rn(p, step);
}

}  // namespace

extern "C" int shorokoo_adam_update_launch(const ShorokooAdamUpdate* update, void* stream) {
    if (update->count == 0) return 0;
    const size_t blocks = (update->count + Threads - 1) / Threads;
    if (blocks > 0x7fffffffu) return static_cast<int>(cudaErrorInvalidConfiguration);
    const auto s = static_cast<cudaStream_t>(stream);
    if (update->decay != nullptr)
        AdamUpdate<true><<<static_cast<unsigned>(blocks), Threads, 0, s>>>(*update);
    else
        AdamUpdate<false><<<static_cast<unsigned>(blocks), Threads, 0, s>>>(*update);
    return static_cast<int>(cudaGetLastError());
}

extern "C" const char* shorokoo_cuda_error_text(int error) {
    return cudaGetErrorString(static_cast<cudaError_t>(error));
}
