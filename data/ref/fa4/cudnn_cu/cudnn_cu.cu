/*
 * cuDNN Flash Attention reference — calls cuDNN 9 SDPA graph API directly.
 *
 * Input/Output: (B, S, H, D) BF16, row-major. No data transformation —
 * cuDNN reads BSHD layout via stride descriptors (zero-copy reinterpret).
 *
 * Compile requires: -I<cudnn>/include -I<cudnn_frontend>/include
 *                   -L<cudnn>/lib -lcudnn -lcuda
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <cudnn_frontend.h>
#include <cstdlib>
#include <cstring>
#include <cstdio>
#include <cmath>
#include <memory>

namespace fe = cudnn_frontend;

/* ── Shared config parsing (same as generated.cu) ──────────────── */

static int env_int(const char *name, int fb) {
    const char *v = getenv(name); return (v && *v) ? atoi(v) : fb;
}
static bool env_bool(const char *name, bool fb) {
    const char *v = getenv(name);
    return v ? (*v == '1' || *v == 't' || *v == 'T') : fb;
}
static int json_int(const char *j, const char *k, int fb) {
    if (!j) return fb;
    const char *p = strstr(j, k);
    if (!p) return fb;
    for (p += strlen(k); *p && (*p=='"'||*p==':'||*p==' '||*p=='\t'); ++p);
    int v = 0; while (*p>='0'&&*p<='9') { v = v*10+(*p-'0'); ++p; }
    return v ? v : fb;
}
static bool json_bool(const char *j, const char *k, bool fb) {
    if (!j) return fb;
    const char *p = strstr(j, k);
    if (!p) return fb;
    for (p += strlen(k); *p && (*p=='"'||*p==':'||*p==' '||*p=='\t'); ++p);
    if (!strncmp(p, "true", 4)) return true;
    if (!strncmp(p, "false", 5)) return false;
    return fb;
}

/* ── cuDNN SDPA graph (cached across calls) ────────────────────── */

static struct {
    cudnnHandle_t handle = nullptr;
    std::shared_ptr<fe::graph::Graph> graph;
    void *workspace = nullptr;
    size_t ws_size = 0;
    int64_t B = 0, S = 0, H = 0, D = 0;
    bool causal = false, ready = false;

    bool build(int64_t b, int64_t s, int64_t h, int64_t d, bool c) {
        if (ready && b==B && s==S && h==H && d==D && c==causal) return true;
        B=b; S=s; H=h; D=d; causal=c;
        if (!handle) cudnnCreate(&handle);

        graph = std::make_shared<fe::graph::Graph>();
        graph->set_io_data_type(fe::DataType_t::BFLOAT16)
              .set_intermediate_data_type(fe::DataType_t::FLOAT)
              .set_compute_data_type(fe::DataType_t::FLOAT);

        /* BSHD physical layout → BHSD logical via strides (zero-copy) */
        int64_t str[4] = {S*H*D, D, H*D, 1}; /* B, H, S, D strides */
        auto mkT = [&](const char* name, int uid) {
            return graph->tensor(fe::graph::Tensor_attributes()
                .set_name(name).set_dim({B,H,S,D}).set_stride({str[0],str[1],str[2],str[3]})
                .set_uid(uid).set_data_type(fe::DataType_t::BFLOAT16));
        };
        auto Q = mkT("Q",1), K = mkT("K",2), V = mkT("V",3);

        auto opts = fe::graph::SDPA_attributes().set_name("sdpa")
            .set_attn_scale(1.0f / sqrtf((float)D)).set_causal_mask(causal);
        auto [O, Stats] = graph->sdpa(Q, K, V, opts);
        (void)Stats;

        O->set_output(true).set_dim({B,H,S,D}).set_stride({str[0],str[1],str[2],str[3]})
          .set_uid(4).set_data_type(fe::DataType_t::BFLOAT16);

        auto e = graph->validate();          if (e.is_bad()) { fprintf(stderr,"validate: %s\n", e.get_message().c_str()); return false; }
        e = graph->build_operation_graph(handle); if (e.is_bad()) { fprintf(stderr,"build: %s\n", e.get_message().c_str()); return false; }
        e = graph->create_execution_plans({fe::HeurMode_t::A}); if (e.is_bad()) { fprintf(stderr,"plans: %s\n", e.get_message().c_str()); return false; }
        e = graph->build_plans(handle);      if (e.is_bad()) { fprintf(stderr,"build_plans: %s\n", e.get_message().c_str()); return false; }

        auto need = graph->get_workspace_size();
        if (need > ws_size) { if (workspace) cudaFree(workspace); cudaMalloc(&workspace, need); ws_size = need; }
        ready = true; return true;
    }

    int run(void *Q, void *K, void *V, void *O, cudaStream_t s) {
        if (!ready) { fprintf(stderr, "cuDNN: not ready\n"); return -1; }
        cudnnSetStream(handle, s);
        std::unordered_map<int64_t,void*> vp = {{1,Q},{2,K},{3,V},{4,O}};
        auto e = graph->execute(handle, vp, workspace);
        if (e.is_bad()) { fprintf(stderr, "cuDNN execute: %s\n", e.get_message().c_str()); return -2; }
        return 0;
    }
} g_sdpa;

/* ── kernel_run — eval harness interface ───────────────────────── */

extern "C" int kernel_run(
    __nv_bfloat16 **inputs, int num_inputs,
    __nv_bfloat16 **outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    const char *cfg = getenv("CUDA_EXEC_CONFIG_JSON");
    int B = env_int("CUDA_EXEC_PARAM_BATCH_SIZE", 0);
    int S = env_int("CUDA_EXEC_PARAM_SEQ_LEN", 0);
    int H = env_int("CUDA_EXEC_PARAM_NUM_HEADS", 0);
    int D = env_int("CUDA_EXEC_PARAM_HEAD_DIM", 0);
    if (!B) B = json_int(cfg, "batch_size", 0);
    if (!S) S = json_int(cfg, "seq_len", 0);
    if (!H) H = json_int(cfg, "num_heads", 16);
    if (!D) D = json_int(cfg, "head_dim", 128);
    bool causal = env_bool("CUDA_EXEC_PARAM_CAUSAL", json_bool(cfg,"causal",false));
    if (!B && !S && n > 0) { B = 1; S = n / (H * D); }
    if (!B || !S) return -1;

    if (!g_sdpa.build(B, S, H, D, causal)) return -3;
    return g_sdpa.run(inputs[0], inputs[1], inputs[2], outputs[0], stream);
}
