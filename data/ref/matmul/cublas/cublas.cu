/*
 * cuBLAS LT BF16 GEMM reference — autotuned (top-20 heuristic candidates).
 *
 * C = A @ B, all BF16, FP32 accumulation.
 * Config shape [M, N], M=N=K (square).
 *
 * On first call per size: requests 20 algorithm candidates from
 * cublasLtMatmulAlgoGetHeuristic, benchmarks each with 3 warmup + 10
 * timed runs, and caches the fastest. Subsequent calls use the winner.
 *
 * Harness contract:
 *   inputs[0]  = A (M×K BF16)
 *   inputs[1]  = B (K×N BF16)
 *   outputs[0] = C (M×N BF16)
 *   n = input_size = M * N
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cublasLt.h>
#include <cstdlib>
#include <cstdio>
#include <cstring>
#include <cmath>
#include <map>
#include <vector>
#include <algorithm>
#include <cfloat>

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

static cublasLtHandle_t g_lt_handle = nullptr;

static cublasLtHandle_t get_lt_handle() {
    if (!g_lt_handle) {
        cublasStatus_t st = cublasLtCreate(&g_lt_handle);
        if (st != CUBLAS_STATUS_SUCCESS) {
            fprintf(stderr, "cublasLtCreate failed: %d\n", (int)st);
            return nullptr;
        }
    }
    return g_lt_handle;
}

/* Per-size cached state: best algorithm + descriptors */
struct CachedState {
    cublasLtMatmulAlgo_t algo;
    size_t workspace_size;
    cublasLtMatmulDesc_t op_desc;
    cublasLtMatrixLayout_t layout_a;
    cublasLtMatrixLayout_t layout_b;
    cublasLtMatrixLayout_t layout_c;
};
static std::map<int, CachedState> g_cache;

/* Workspace */
static void *g_workspace = nullptr;
static size_t g_workspace_size = 0;
static const size_t MAX_WORKSPACE = 32 * 1024 * 1024; /* 32 MB */

static void ensure_workspace() {
    if (!g_workspace) {
        cudaMalloc(&g_workspace, MAX_WORKSPACE);
        g_workspace_size = MAX_WORKSPACE;
    }
}

/* ── Autotune: benchmark top-N candidates, return best ──────────── */

static const int AUTOTUNE_CANDIDATES = 20;
static const int AUTOTUNE_WARMUP     = 3;
static const int AUTOTUNE_TRIALS     = 10;

static CachedState create_and_autotune(
    cublasLtHandle_t handle,
    __nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C,
    int M, int N, int K, cudaStream_t stream)
{
    ensure_workspace();

    CachedState state;

    cublasLtMatmulDescCreate(&state.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);
    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(state.op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &op_n, sizeof(op_n));
    cublasLtMatmulDescSetAttribute(state.op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &op_n, sizeof(op_n));

    /* Row-major C = A @ B → col-major C^T = B^T @ A^T */
    cublasLtMatrixLayoutCreate(&state.layout_a, CUDA_R_16BF, N, K, N);
    cublasLtMatrixLayoutCreate(&state.layout_b, CUDA_R_16BF, K, M, K);
    cublasLtMatrixLayoutCreate(&state.layout_c, CUDA_R_16BF, N, M, N);

    /* Get top-20 heuristic candidates */
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &g_workspace_size, sizeof(g_workspace_size));

    cublasLtMatmulHeuristicResult_t results[AUTOTUNE_CANDIDATES];
    int num_returned = 0;

    cublasLtMatmulAlgoGetHeuristic(handle, state.op_desc,
        state.layout_a, state.layout_b, state.layout_c, state.layout_c,
        pref, AUTOTUNE_CANDIDATES, results, &num_returned);

    cublasLtMatmulPreferenceDestroy(pref);

    if (num_returned < 1) {
        fprintf(stderr, "cublasLtMatmulAlgoGetHeuristic returned 0 candidates\n");
        state.algo = {};
        state.workspace_size = 0;
        return state;
    }

    /* Benchmark each candidate */
    float alpha = 1.0f, beta = 0.0f;
    cudaEvent_t start_ev, stop_ev;
    cudaEventCreate(&start_ev);
    cudaEventCreate(&stop_ev);

    int best_idx = 0;
    float best_ms = FLT_MAX;

    for (int c = 0; c < num_returned; c++) {
        /* Verify algorithm works */
        cublasStatus_t st = cublasLtMatmul(handle, state.op_desc,
            &alpha, B, state.layout_a, A, state.layout_b,
            &beta, C, state.layout_c, C, state.layout_c,
            &results[c].algo, g_workspace, results[c].workspaceSize, stream);
        cudaStreamSynchronize(stream);
        if (st != CUBLAS_STATUS_SUCCESS) continue;

        /* Warmup */
        for (int w = 0; w < AUTOTUNE_WARMUP; w++) {
            cublasLtMatmul(handle, state.op_desc,
                &alpha, B, state.layout_a, A, state.layout_b,
                &beta, C, state.layout_c, C, state.layout_c,
                &results[c].algo, g_workspace, results[c].workspaceSize, stream);
        }
        cudaStreamSynchronize(stream);

        /* Timed runs */
        cudaEventRecord(start_ev, stream);
        for (int t = 0; t < AUTOTUNE_TRIALS; t++) {
            cublasLtMatmul(handle, state.op_desc,
                &alpha, B, state.layout_a, A, state.layout_b,
                &beta, C, state.layout_c, C, state.layout_c,
                &results[c].algo, g_workspace, results[c].workspaceSize, stream);
        }
        cudaEventRecord(stop_ev, stream);
        cudaEventSynchronize(stop_ev);

        float elapsed_ms = 0.0f;
        cudaEventElapsedTime(&elapsed_ms, start_ev, stop_ev);
        float avg_ms = elapsed_ms / AUTOTUNE_TRIALS;

        if (avg_ms < best_ms) {
            best_ms = avg_ms;
            best_idx = c;
        }
    }

    cudaEventDestroy(start_ev);
    cudaEventDestroy(stop_ev);

    state.algo = results[best_idx].algo;
    state.workspace_size = results[best_idx].workspaceSize;

    fprintf(stderr, "[ref-cublas] autotune %dx%d: %d/%d candidates, best=#%d (%.3f ms)\n",
            M, N, num_returned, AUTOTUNE_CANDIDATES, best_idx, best_ms);

    return state;
}

/* ── kernel_run — harness interface ───────────────────────────────── */

extern "C" int kernel_run(
    __nv_bfloat16 **inputs,  int num_inputs,
    __nv_bfloat16 **outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    if (num_inputs < 2 || num_outputs < 1) return -1;

    int M = 0, N = 0;

    const char *shape_env = getenv("CUDA_EXEC_PARAM_SHAPE");
    if (shape_env && *shape_env) {
        const char *p = shape_env;
        while (*p == '[' || *p == ' ') ++p;
        while (*p >= '0' && *p <= '9') { M = M * 10 + (*p - '0'); ++p; }
        while (*p == ',' || *p == ' ') ++p;
        while (*p >= '0' && *p <= '9') { N = N * 10 + (*p - '0'); ++p; }
    }

    if (M == 0 || N == 0) {
        const char *config_json = getenv("CUDA_EXEC_CONFIG_JSON");
        json_shape(config_json, &M, &N);
    }

    if ((M == 0 || N == 0) && n > 0) {
        M = (int)sqrtf((float)n);
        N = M;
    }

    if (M == 0 || N == 0) return -2;

    int K = M;

    __nv_bfloat16 *A = inputs[0];
    __nv_bfloat16 *B = inputs[1];
    __nv_bfloat16 *C = outputs[0];

    cublasLtHandle_t handle = get_lt_handle();
    if (!handle) return -3;
    ensure_workspace();

    /* Autotune on first call per size, cache the winner */
    if (g_cache.find(M) == g_cache.end()) {
        g_cache[M] = create_and_autotune(handle, A, B, C, M, N, K, stream);
    }

    CachedState &s = g_cache[M];

    float alpha = 1.0f, beta = 0.0f;

    cublasStatus_t st = cublasLtMatmul(handle, s.op_desc,
        &alpha, B, s.layout_a, A, s.layout_b,
        &beta, C, s.layout_c, C, s.layout_c,
        &s.algo, g_workspace, s.workspace_size, stream);

    if (st != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasLtMatmul failed: %d\n", (int)st);
        return -4;
    }

    return 0;
}
