/*
 * cuDNN Flash Attention reference — calls cuDNN 9 graph API directly.
 *
 * Uses cudnn_frontend (header-only C++ library) to build an SDPA graph,
 * then executes cuDNN's fused flash attention kernel on H100.
 *
 * Input layout:  Q, K, V each (B, S, H, D) — BF16, row-major
 * Output layout: O same shape — BF16
 *
 * Compile: nvcc -std=c++17 -O3 -gencode arch=compute_90a,code=sm_90a
 *          -I<cudnn_frontend>/include -I<cudnn>/include
 *          -lcuda -lcudnn
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <memory>

#include <cudnn_frontend.h>

namespace fe = cudnn_frontend;

/* ── Helpers ───────────────────────────────────────────────────────── */

__device__ __host__ constexpr
int cdiv(int a, int b) { return (a + b - 1) / b; }

static int parse_int_env(const char *name, int fallback) {
    const char *v = getenv(name);
    if (v && *v) return atoi(v);
    return fallback;
}

static bool parse_bool_env(const char *name, bool fallback) {
    const char *v = getenv(name);
    if (!v) return fallback;
    return (strcmp(v, "true") == 0 || strcmp(v, "1") == 0 ||
            strcmp(v, "True") == 0 || strcmp(v, "TRUE") == 0);
}

static int json_int(const char *json, const char *key, int fallback) {
    if (!json) return fallback;
    const char *p = strstr(json, key);
    if (!p) return fallback;
    p += strlen(key);
    while (*p && (*p == '"' || *p == ':' || *p == ' ' || *p == '\t')) ++p;
    if (*p < '0' || *p > '9') return fallback;
    int val = 0;
    while (*p >= '0' && *p <= '9') { val = val * 10 + (*p - '0'); ++p; }
    return val;
}

static bool json_bool(const char *json, const char *key, bool fallback) {
    if (!json) return fallback;
    const char *p = strstr(json, key);
    if (!p) return fallback;
    p += strlen(key);
    while (*p && (*p == '"' || *p == ':' || *p == ' ' || *p == '\t')) ++p;
    if (strncmp(p, "true", 4) == 0) return true;
    if (strncmp(p, "false", 5) == 0) return false;
    return fallback;
}

/* ── Persistent cuDNN state ─────────────────────────────────────── */

struct CudnnSDPA {
    cudnnHandle_t handle = nullptr;
    std::shared_ptr<fe::graph::Graph> graph;
    int64_t B = 0, S = 0, H = 0, D = 0;
    bool causal = false;
    bool built = false;

    /* Tensor UIDs for variant pack */
    static constexpr int64_t Q_UID = 1;
    static constexpr int64_t K_UID = 2;
    static constexpr int64_t V_UID = 3;
    static constexpr int64_t O_UID = 4;
    static constexpr int64_t STATS_UID = 5;

    /* Workspace */
    void *workspace = nullptr;
    size_t workspace_size = 0;

    /* Stats tensor (softmax statistics, required by cuDNN) */
    void *stats_ptr = nullptr;

    ~CudnnSDPA() {
        if (workspace) cudaFree(workspace);
        if (stats_ptr) cudaFree(stats_ptr);
        if (handle) cudnnDestroy(handle);
    }

    bool build(int64_t b, int64_t s, int64_t h, int64_t d, bool is_causal) {
        if (built && b == B && s == S && h == H && d == D && is_causal == causal)
            return true;

        B = b; S = s; H = h; D = d; causal = is_causal;

        if (!handle) {
            auto status = cudnnCreate(&handle);
            if (status != CUDNN_STATUS_SUCCESS) {
                fprintf(stderr, "cudnnCreate failed: %d\n", (int)status);
                return false;
            }
        }

        graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::BFLOAT16)
              .set_intermediate_data_type(fe::DataType_t::FLOAT)
              .set_compute_data_type(fe::DataType_t::FLOAT);

        /* Input tensors: (B, H, S, D) — cuDNN expects BHSD layout
         * Our data is (B, S, H, D) row-major, so we pass strides
         * to reinterpret as (B, H, S, D) without transposing. */
        int64_t q_stride[4] = {S * H * D, D, H * D, 1};  /* BHSD from BSHD */
        int64_t o_stride[4] = {S * H * D, D, H * D, 1};

        auto Q_t = graph->tensor(fe::graph::Tensor_attributes()
            .set_name("Q")
            .set_dim({B, H, S, D})
            .set_stride({q_stride[0], q_stride[1], q_stride[2], q_stride[3]})
            .set_uid(Q_UID)
            .set_data_type(fe::DataType_t::BFLOAT16));

        auto K_t = graph->tensor(fe::graph::Tensor_attributes()
            .set_name("K")
            .set_dim({B, H, S, D})
            .set_stride({q_stride[0], q_stride[1], q_stride[2], q_stride[3]})
            .set_uid(K_UID)
            .set_data_type(fe::DataType_t::BFLOAT16));

        auto V_t = graph->tensor(fe::graph::Tensor_attributes()
            .set_name("V")
            .set_dim({B, H, S, D})
            .set_stride({q_stride[0], q_stride[1], q_stride[2], q_stride[3]})
            .set_uid(V_UID)
            .set_data_type(fe::DataType_t::BFLOAT16));

        /* SDPA attributes */
        float attn_scale = 1.0f / sqrtf(static_cast<float>(D));
        auto sdpa_opts = fe::graph::SDPA_attributes()
            .set_name("sdpa")
            .set_is_inference(true)
            .set_attn_scale(attn_scale)
            .set_causal_mask(is_causal);

        auto [O_t, Stats_t] = graph->sdpa(Q_t, K_t, V_t, sdpa_opts);

        O_t->set_output(true)
            .set_dim({B, H, S, D})
            .set_stride({o_stride[0], o_stride[1], o_stride[2], o_stride[3]})
            .set_uid(O_UID)
            .set_data_type(fe::DataType_t::BFLOAT16);

        /* Stats_t is not needed for inference — do not set as output */
        (void)Stats_t;

        /* Validate + build */
        auto err = graph->validate();
        if (err.is_bad()) {
            fprintf(stderr, "graph validate failed: %s\n", err.get_message().c_str());
            return false;
        }

        err = graph->build_operation_graph(handle);
        if (err.is_bad()) {
            fprintf(stderr, "build_operation_graph failed: %s\n", err.get_message().c_str());
            return false;
        }

        err = graph->create_execution_plans({fe::HeurMode_t::A});
        if (err.is_bad()) {
            fprintf(stderr, "create_execution_plans failed: %s\n", err.get_message().c_str());
            return false;
        }

        err = graph->build_plans(handle);
        if (err.is_bad()) {
            fprintf(stderr, "build_plans failed: %s\n", err.get_message().c_str());
            return false;
        }

        /* Workspace */
        auto new_ws_size = graph->get_workspace_size();
        if (new_ws_size > workspace_size) {
            if (workspace) cudaFree(workspace);
            cudaMalloc(&workspace, new_ws_size);
            workspace_size = new_ws_size;
        }

        /* Stats buffer */
        size_t stats_size = B * H * S * sizeof(float);
        if (stats_ptr) cudaFree(stats_ptr);
        cudaMalloc(&stats_ptr, stats_size);

        built = true;
        return true;
    }

    int execute(void *Q_ptr, void *K_ptr, void *V_ptr, void *O_ptr,
                cudaStream_t stream) {
        if (!built) return -1;

        cudnnSetStream(handle, stream);

        std::unordered_map<int64_t, void*> variant_pack = {
            {Q_UID, Q_ptr},
            {K_UID, K_ptr},
            {V_UID, V_ptr},
            {O_UID, O_ptr},
        };

        auto err = graph->execute(handle, variant_pack, workspace);
        if (err.is_bad()) {
            fprintf(stderr, "graph execute failed: %s\n", err.get_message().c_str());
            return -2;
        }
        return 0;
    }
};

static CudnnSDPA g_sdpa;

/* ── kernel_run — harness interface ───────────────────────────────── */

extern "C" int kernel_run(
    __nv_bfloat16 **inputs,  int num_inputs,
    __nv_bfloat16 **outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    const char *config_json = getenv("CUDA_EXEC_CONFIG_JSON");

    int B = parse_int_env("CUDA_EXEC_PARAM_BATCH_SIZE", 0);
    int S = parse_int_env("CUDA_EXEC_PARAM_SEQ_LEN",    0);
    int H = parse_int_env("CUDA_EXEC_PARAM_NUM_HEADS",  0);
    int D = parse_int_env("CUDA_EXEC_PARAM_HEAD_DIM",   0);

    if (B == 0) B = json_int(config_json, "batch_size", 0);
    if (S == 0) S = json_int(config_json, "seq_len",    0);
    if (H == 0) H = json_int(config_json, "num_heads",  0);
    if (D == 0) D = json_int(config_json, "head_dim",   0);

    if (D == 0) D = 128;
    if (H == 0) H = 16;
    if (S == 0 && B == 0 && n > 0) {
        int total_tokens = n / (H * D);
        B = 1;
        S = total_tokens / B;
    }
    if (B == 0 || S == 0) return -1;

    bool causal = parse_bool_env("CUDA_EXEC_PARAM_CAUSAL", false);
    if (!causal && config_json)
        causal = json_bool(config_json, "causal", false);

    /* Build/cache the cuDNN graph */
    if (!g_sdpa.build(B, S, H, D, causal)) return -3;

    /* Execute */
    return g_sdpa.execute(inputs[0], inputs[1], inputs[2], outputs[0], stream);
}
