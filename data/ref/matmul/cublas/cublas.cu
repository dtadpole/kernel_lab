/*
 * cuBLAS LT BF16 GEMM reference — PyTorch-matching configuration.
 *
 * C = A @ B, all BF16, FP32 accumulation.
 * Uses the standard row-major cuBLAS trick: C = A@B → C^T = B^T @ A^T
 *   - TRANSA=N, TRANSB=N, swap A↔B pointers (PyTorch convention)
 *   - Alignment hints for A, B, C pointers
 *   - Heuristic #1 (no manual autotune — cuBLAS heuristic is accurate)
 *   - Workspace from cudaMalloc
 *
 * Harness contract:
 *   inputs[0]  = A (M×K BF16, row-major)
 *   inputs[1]  = B (K×N BF16, row-major)
 *   outputs[0] = C (M×N BF16, row-major)
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

/* Per-size cached state */
struct CachedState {
    cublasLtMatmulAlgo_t algo;
    size_t workspace_size;
    cublasLtMatmulDesc_t op_desc;
    cublasLtMatrixLayout_t layout_a;
    cublasLtMatrixLayout_t layout_b;
    cublasLtMatrixLayout_t layout_c;
};
static std::map<int, CachedState> g_cache;

/* Workspace — 32 MB (matches PyTorch's default) */
static void *g_workspace = nullptr;
static size_t g_workspace_size = 0;
static const size_t MAX_WORKSPACE = 32 * 1024 * 1024;

static void ensure_workspace() {
    if (!g_workspace) {
        cudaMalloc(&g_workspace, MAX_WORKSPACE);
        g_workspace_size = MAX_WORKSPACE;
    }
}

/* Pointer alignment (matches PyTorch's _getAlignment) */
static uint32_t get_alignment(uintptr_t ptr) {
    /* Returns the largest power-of-2 that divides ptr, capped at 256 */
    for (int shift = 8; shift > 0; shift--) {
        if (ptr % (1u << shift) == 0) return (1u << shift);
    }
    return 1;
}

/* ── Setup: create descriptors + pick heuristic #1 (PyTorch style) ── */

static CachedState create_cached(
    cublasLtHandle_t handle,
    __nv_bfloat16 *A, __nv_bfloat16 *B, __nv_bfloat16 *C,
    int M, int N, int K, cudaStream_t stream)
{
    ensure_workspace();

    CachedState state;

    /* Compute descriptor: FP32 accumulation */
    cublasLtMatmulDescCreate(&state.op_desc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

    /* Row-major C = A @ B via the standard cuBLAS col-major trick:
     *   C^T = B^T @ A^T  (col-major)
     * Row-major data is col-major transposed, so no explicit transposes needed:
     *   TRANSA=N (first matrix = B), TRANSB=N (second matrix = A)
     *   cuBLAS "A" desc = B as col-major (N, K) with ld=N
     *   cuBLAS "B" desc = A as col-major (K, M) with ld=K
     *   cuBLAS "C" desc = C^T as col-major (N, M) with ld=N
     * The col-major (N, M) output is exactly row-major (M, N) — correct. */
    cublasOperation_t op_n = CUBLAS_OP_N;
    cublasLtMatmulDescSetAttribute(state.op_desc, CUBLASLT_MATMUL_DESC_TRANSA,
                                   &op_n, sizeof(op_n));
    cublasLtMatmulDescSetAttribute(state.op_desc, CUBLASLT_MATMUL_DESC_TRANSB,
                                   &op_n, sizeof(op_n));

    /* Matrix layouts — "A" in cuBLAS = our B, "B" in cuBLAS = our A */
    int lda = N;  /* cuBLAS "A" = B: col-major (N, K), ld=N */
    int ldb = K;  /* cuBLAS "B" = A: col-major (K, M), ld=K */
    int ldc = N;  /* cuBLAS "C" = C^T: col-major (N, M), ld=N */

    cublasLtMatrixLayoutCreate(&state.layout_a, CUDA_R_16BF, N, K, lda);
    cublasLtMatrixLayoutCreate(&state.layout_b, CUDA_R_16BF, K, M, ldb);
    cublasLtMatrixLayoutCreate(&state.layout_c, CUDA_R_16BF, N, M, ldc);

    /* Preference with alignment hints (matches PyTorch) */
    cublasLtMatmulPreference_t pref;
    cublasLtMatmulPreferenceCreate(&pref);
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,
        &g_workspace_size, sizeof(g_workspace_size));

    /* Alignment hints — cuBLAS "A" = our B, cuBLAS "B" = our A */
    uint32_t align_a = get_alignment(reinterpret_cast<uintptr_t>(B));  /* cuBLAS "A" = B */
    uint32_t align_b = get_alignment(reinterpret_cast<uintptr_t>(A));  /* cuBLAS "B" = A */
    uint32_t align_c = get_alignment(reinterpret_cast<uintptr_t>(C));
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_A_BYTES, &align_a, sizeof(align_a));
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_B_BYTES, &align_b, sizeof(align_b));
    cublasLtMatmulPreferenceSetAttribute(pref,
        CUBLASLT_MATMUL_PREF_MIN_ALIGNMENT_C_BYTES, &align_c, sizeof(align_c));

    /* Get heuristic #1 — PyTorch uses requestedAlgoCount=1 */
    cublasLtMatmulHeuristicResult_t heuristic;
    int num_returned = 0;

    cublasLtMatmulAlgoGetHeuristic(handle, state.op_desc,
        state.layout_a, state.layout_b, state.layout_c, state.layout_c,
        pref, 1, &heuristic, &num_returned);

    cublasLtMatmulPreferenceDestroy(pref);

    if (num_returned < 1) {
        fprintf(stderr, "cublasLtMatmulAlgoGetHeuristic returned 0 candidates\n");
        state.algo = {};
        state.workspace_size = 0;
        return state;
    }

    state.algo = heuristic.algo;
    state.workspace_size = heuristic.workspaceSize;

    fprintf(stderr, "[ref-cublas] heuristic %dx%d: workspace=%zu\n",
            M, N, state.workspace_size);

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

    /* Cache per size — heuristic #1 (no autotune needed) */
    if (g_cache.find(M) == g_cache.end()) {
        g_cache[M] = create_cached(handle, A, B, C, M, N, K, stream);
    }

    CachedState &s = g_cache[M];

    float alpha = 1.0f, beta = 0.0f;

    /* Row-major trick: swap A↔B in cuBLAS call.
     * cuBLAS "A" = our B, cuBLAS "B" = our A → C^T = B^T @ A^T = (A@B)^T */
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
