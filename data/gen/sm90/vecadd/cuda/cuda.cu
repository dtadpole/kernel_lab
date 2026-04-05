/*
 * BF16 vector-add kernel for cuda_exec evaluation.
 *
 * Implements the kernel_run contract:
 *   extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
 *                             __nv_bfloat16** outputs, int num_outputs,
 *                             int n, cudaStream_t stream);
 */
#include <cuda_bf16.h>

__global__ void vector_add(const __nv_bfloat16* a,
                           const __nv_bfloat16* b,
                           __nv_bfloat16* out,
                           int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = __float2bfloat16(__bfloat162float(a[i]) + __bfloat162float(b[i]));
    }
}

extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
                          __nv_bfloat16** outputs, int num_outputs,
                          int n, cudaStream_t stream) {
    const __nv_bfloat16* a = inputs[0];
    const __nv_bfloat16* b = inputs[1];
    __nv_bfloat16* out = outputs[0];
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;
    vector_add<<<blocks, threads, 0, stream>>>(a, b, out, n);
    return 0;
}
