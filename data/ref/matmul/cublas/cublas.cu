/*
 * cuBLAS BF16 GEMM reference — with per-size algorithm autotuning.
 *
 * C = A @ B, all BF16, FP32 accumulation.
 * Config shape [M, N], M=N=K (square).
 *
 * On first call for each matrix size, tries all cuBLAS algorithms
 * and caches the fastest. Subsequent calls use the cached best.
 *
 * Harness contract:
 *   inputs[0]  = A (M×K BF16)
 *   inputs[1]  = B (K×N BF16)
 *   outputs[0] = C (M×N BF16)
 *   n = input_size = M * N
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <map>

/* ── Helpers ───────────────────────────────────────────────────────── */

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

/* ── Persistent state ─────────────────────────────────────────────── */

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

/* Per-size cached best algorithm */
static std::map<int, cublasGemmAlgo_t> g_best_algo;

static cublasGemmAlgo_t autotune(
    cublasHandle_t handle, cudaStream_t stream,
    int M, int N, int K,
    __nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C)
{
    float alpha = 1.0f, beta = 0.0f;
    cublasGemmAlgo_t best = CUBLAS_GEMM_DEFAULT_TENSOR_OP;
    float best_time = 1e9f;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    /* Algorithms to try: standard (0-23) + tensor op (99-115) */
    int algos[] = {
        0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11,
        12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23,
        99, 100, 101, 102, 103, 104, 105, 106, 107, 108, 109, 110,
        111, 112, 113, 114, 115
    };
    int num_algos = sizeof(algos) / sizeof(algos[0]);
    int num_timing_runs = 3;

    for (int ai = 0; ai < num_algos; ai++) {
        cublasGemmAlgo_t algo = (cublasGemmAlgo_t)algos[ai];

        /* Warmup — skip if algorithm doesn't support this config */
        cublasStatus_t st = cublasGemmEx(
            handle, CUBLAS_OP_N, CUBLAS_OP_N,
            N, M, K, &alpha,
            B, CUDA_R_16BF, N,
            A, CUDA_R_16BF, K,
            &beta,
            C, CUDA_R_16BF, N,
            CUBLAS_COMPUTE_32F, algo);

        if (st != CUBLAS_STATUS_SUCCESS) continue;
        cudaStreamSynchronize(stream);

        /* Time it */
        cudaEventRecord(start, stream);
        for (int r = 0; r < num_timing_runs; r++) {
            cublasGemmEx(
                handle, CUBLAS_OP_N, CUBLAS_OP_N,
                N, M, K, &alpha,
                B, CUDA_R_16BF, N,
                A, CUDA_R_16BF, K,
                &beta,
                C, CUDA_R_16BF, N,
                CUBLAS_COMPUTE_32F, algo);
        }
        cudaEventRecord(stop, stream);
        cudaEventSynchronize(stop);

        float ms = 0;
        cudaEventElapsedTime(&ms, start, stop);
        ms /= num_timing_runs;

        if (ms < best_time) {
            best_time = ms;
            best = algo;
        }
    }

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    fprintf(stderr, "cuBLAS autotune M=%d: best algo=%d (%.3f ms)\n",
            M, (int)best, best_time);
    return best;
}

/* ── kernel_run — harness interface ───────────────────────────────── */

extern "C" int kernel_run(
    __nv_bfloat16 **inputs,  int num_inputs,
    __nv_bfloat16 **outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    if (num_inputs < 2 || num_outputs < 1) return -1;

    const char *config_json = getenv("CUDA_EXEC_CONFIG_JSON");

    int M = 0, N = 0;

    const char *shape_env = getenv("CUDA_EXEC_PARAM_SHAPE");
    if (shape_env && *shape_env) {
        const char *p = shape_env;
        while (*p == '[' || *p == ' ') ++p;
        while (*p >= '0' && *p <= '9') { M = M * 10 + (*p - '0'); ++p; }
        while (*p == ',' || *p == ' ') ++p;
        while (*p >= '0' && *p <= '9') { N = N * 10 + (*p - '0'); ++p; }
    }

    if (M == 0 || N == 0)
        json_shape(config_json, &M, &N);

    if ((M == 0 || N == 0) && n > 0) {
        M = (int)sqrtf((float)n);
        N = M;
    }

    if (M == 0 || N == 0) return -2;

    int K = M;

    __nv_bfloat16 *A = inputs[0];
    __nv_bfloat16 *B = inputs[1];
    __nv_bfloat16 *C = outputs[0];

    cublasHandle_t handle = get_handle();
    if (!handle) return -3;

    cublasSetStream(handle, stream);

    /* Autotune on first call per size */
    if (g_best_algo.find(M) == g_best_algo.end()) {
        g_best_algo[M] = autotune(handle, stream, M, N, K, A, B, C);
    }

    float alpha = 1.0f;
    float beta  = 0.0f;

    cublasStatus_t st = cublasGemmEx(
        handle,
        CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        B, CUDA_R_16BF, N,
        A, CUDA_R_16BF, K,
        &beta,
        C, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F,
        g_best_algo[M]
    );

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasGemmEx failed: %d\n", (int)st);
        return -4;
    }

    return 0;
}
