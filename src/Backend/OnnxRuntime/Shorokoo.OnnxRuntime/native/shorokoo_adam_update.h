// The one fused Adam or AdamW update the CUDA build of shorokoo_ort_ops.cpp hands
// shorokoo_adam_update.cu: every pointer is device memory, the coefficients included, so that
// launching it reads nothing back to the host and the launch waits for nothing.
//
// Plain C types only: the operator's side is compiled by the host compiler with no CUDA header,
// and the kernel's side by nvcc.
#ifndef SHOROKOO_ADAM_UPDATE_H
#define SHOROKOO_ADAM_UPDATE_H

#include <stddef.h>

struct ShorokooAdamUpdate {
    const float* p;
    const float* m;
    const float* v;
    const float* g;
    float* pOut;
    float* mOut;
    float* vOut;
    // One element each: beta1, c1, beta2, c2, eps, step, and the decay or null where there is none.
    const float* beta1;
    const float* c1;
    const float* beta2;
    const float* c2;
    const float* eps;
    const float* step;
    const float* decay;
    size_t count;
};

#ifdef __cplusplus
extern "C" {
#endif

// Queues the update on `stream` (a cudaStream_t) and returns at once: 0, or the CUDA error the
// launch met.
int shorokoo_adam_update_launch(const struct ShorokooAdamUpdate* update, void* stream);

// The text of a CUDA error `shorokoo_adam_update_launch` returned.
const char* shorokoo_cuda_error_text(int error);

#ifdef __cplusplus
}
#endif

#endif
