/*
 * cuBLAS BF16 GEMM reference — calls cublasGemmEx directly from C++.
 *
 * Square matrix multiplication: C = A @ B
 * A (M, K), B (K, N), C (M, N) — all BF16, FP32 accumulation.
 *
 * Config shape is [M, N] with M=N=K (square).
 * Harness provides:
 *   inputs[0] = A (input_size elements BF16)
 *   inputs[1] = B (input_size elements BF16)
 *   outputs[0] = C (input_size elements BF16)
 *   n = input_size = M * N
 *
 * Compile: nvcc -std=c++17 -O3 -gencode arch=compute_90a,code=sm_90a
 *          -lcublas
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>

/* ── Helpers ───────────────────────────────────────────────────────── */

/* Parse shape array [M, N] from JSON.  Returns M and N via pointers. */
static bool json_shape(const char *json, int *M, int *N) {
    if (!json) return false;
    const char *p = strstr(json, "\"shape\"");
    if (!p) return false;
    p = strchr(p, '[');
    if (!p) return false;
    ++p;
    while (*p == ' ') ++p;
    *M = 0;
    while (*p >= '0' && *p <= '9') { *M = *M * 10 + (*p - '0'); ++p; }
    while (*p == ',' || *p == ' ') ++p;
    *N = 0;
    while (*p >= '0' && *p <= '9') { *N = *N * 10 + (*p - '0'); ++p; }
    return (*M > 0 && *N > 0);
}

/* ── Persistent cuBLAS state ──────────────────────────────────────── */

static cublasHandle_t g_handle = nullptr;

static cublasHandle_t get_handle() {
    if (!g_handle) {
        cublasStatus_t st = cublasCreate(&g_handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasCreate failed: %d\n", (int)st);
            return nullptr;
        }
    }
    return g_handle;
}

/* ── kernel_run — harness interface ───────────────────────────────── */

extern "C" int kernel_run(
    __nv_bfloat16 **inputs,  int num_inputs,
    __nv_bfloat16 **outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    if (num_inputs < 2 || num_outputs < 1) return -1;

    const char *config_json = getenv("CUDA_EXEC_CONFIG_JSON");

    /* Determine M, N from config shape or env vars */
    int M = 0, N = 0;

    /* Try CUDA_EXEC_PARAM_SHAPE env (JSON array string like "[4096, 4096]") */
    const char *shape_env = getenv("CUDA_EXEC_PARAM_SHAPE");
    if (shape_env && *shape_env) {
        const char *p = shape_env;
        while (*p == '[' || *p == ' ') ++p;
        while (*p >= '0' && *p <= '9') { M = M * 10 + (*p - '0'); ++p; }
        while (*p == ',' || *p == ' ') ++p;
        while (*p >= '0' && *p <= '9') { N = N * 10 + (*p - '0'); ++p; }
    }

    /* Fall back to config JSON */
    if (M == 0 || N == 0)
        json_shape(config_json, &M, &N);

    /* Fall back to sqrt(n) for square matrices */
    if ((M == 0 || N == 0) && n > 0) {
        M = (int)sqrtf((float)n);
        N = M;
    }

    if (M == 0 || N == 0) return -2;

    /* Square matmul: K = M */
    int K = M;

    __nv_bfloat16 *A = inputs[0];   /* (M, K) row-major */
    __nv_bfloat16 *B = inputs[1];   /* (K, N) row-major */
    __nv_bfloat16 *C = outputs[0];  /* (M, N) row-major */

    cublasHandle_t handle = get_handle();
    if (!handle) return -3;

    cublasSetStream(handle, stream);

    /*
     * cuBLAS uses column-major by default.
     * For row-major C = A @ B, we compute C^T = B^T @ A^T in col-major.
     *
     * Col-major view:
     *   C^T (N x M) = B^T (N x K) @ A^T (K x M)
     *   m=N, n=M, k=K
     *   A_cublas = B_ptr (lda=N)
     *   B_cublas = A_ptr (ldb=K)
     *   C_cublas = C_ptr (ldc=N)
     */
    float alpha = 1.0f;
    float beta  = 0.0f;

    cublasStatus_t st = cublasGemmEx(
        handle,
        CUBLAS_OP_N,            /* transa */
        CUBLAS_OP_N,            /* transb */
        N, M, K,                /* m, n, k */
        &alpha,
        B, CUDA_R_16BF, N,      /* A_cublas = B, lda = N */
        A, CUDA_R_16BF, K,      /* B_cublas = A, ldb = K */
        &beta,
        C, CUDA_R_16BF, N,      /* C_cublas = C, ldc = N */
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    );

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasGemmEx failed: %d\n", (int)st);
        return -4;
    }

    return 0;
}
