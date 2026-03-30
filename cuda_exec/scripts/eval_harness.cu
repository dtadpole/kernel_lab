/*
 * eval_harness.cu — BF16-only evaluation harness for cuda_exec.
 *
 * All input and output buffers use __nv_bfloat16.  Kernel authors implement
 * a single function with this exact signature:
 *
 *   #include <cuda_bf16.h>
 *
 *   extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
 *                             __nv_bfloat16** outputs, int num_outputs,
 *                             int n, cudaStream_t stream);
 *
 * No custom headers or structs are required.  The only include needed is
 * <cuda_bf16.h>, which is part of the CUDA standard toolkit.
 *
 * The harness provides: main(), env-based config, BF16 input generation,
 * CUDA event timing, warmup, and structured JSON output on stdout.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

/* ------------------------------------------------------------------ */
/* Kernel contract — implemented by the kernel .cu file               */
/* ------------------------------------------------------------------ */
extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
                          __nv_bfloat16** outputs, int num_outputs,
                          int n, cudaStream_t stream);

/* ------------------------------------------------------------------ */
/* Internal harness config (not exposed to kernel authors)            */
/* ------------------------------------------------------------------ */
struct HarnessConfig {
    char config_slug[256];
    char config_json[8192];
    int  input_size;
    int  rank;
    char shape_kind[64];
    char shape_json[1024];
    int  num_warmups;
    int  num_trials;
};

/* ------------------------------------------------------------------ */
/* Env helpers                                                        */
/* ------------------------------------------------------------------ */
static const char* env_or(const char* name, const char* fallback) {
    const char* v = getenv(name);
    return (v && v[0]) ? v : fallback;
}

static int env_int(const char* name, int fallback) {
    const char* v = getenv(name);
    return v ? atoi(v) : fallback;
}

/* ------------------------------------------------------------------ */
/* Parse config from environment variables                            */
/* ------------------------------------------------------------------ */
static void parse_config(HarnessConfig* cfg) {
    memset(cfg, 0, sizeof(*cfg));

    strncpy(cfg->config_slug,
            env_or("CUDA_EXEC_CONFIG_ID", "default"),
            sizeof(cfg->config_slug) - 1);

    strncpy(cfg->config_json,
            env_or("CUDA_EXEC_CONFIG_JSON", "{}"),
            sizeof(cfg->config_json) - 1);

    cfg->input_size = env_int("CUDA_EXEC_PARAM_INPUT_SIZE",
                     env_int("CUDA_EXEC_EXTRA_INPUT_SIZE", 1 << 20));

    cfg->rank = env_int("CUDA_EXEC_PARAM_RANK",
               env_int("CUDA_EXEC_EXTRA_RANK", 1));

    strncpy(cfg->shape_kind,
            env_or("CUDA_EXEC_PARAM_SHAPE_KIND",
                   env_or("CUDA_EXEC_EXTRA_SHAPE_KIND", "1d")),
            sizeof(cfg->shape_kind) - 1);

    strncpy(cfg->shape_json,
            env_or("CUDA_EXEC_PARAM_SHAPE",
                   env_or("CUDA_EXEC_EXTRA_SHAPE", "[1048576]")),
            sizeof(cfg->shape_json) - 1);

    cfg->num_warmups = env_int("CUDA_EXEC_NUM_WARMUPS", 5);
    cfg->num_trials  = env_int("CUDA_EXEC_NUM_TRIALS", 10);
}

/* ------------------------------------------------------------------ */
/* Generate deterministic BF16 input data (arange pattern)            */
/* ------------------------------------------------------------------ */
static void fill_arange(__nv_bfloat16* host_buf, int count) {
    for (int i = 0; i < count; i++) {
        host_buf[i] = __float2bfloat16(static_cast<float>(i));
    }
}

/* ------------------------------------------------------------------ */
/* JSON output (BF16 values converted to float for printing)          */
/* ------------------------------------------------------------------ */
static void print_json(const HarnessConfig* cfg,
                       const std::vector<double>& latencies,
                       const std::vector<std::vector<__nv_bfloat16>>& outputs) {
    /* Latency stats */
    std::vector<double> sorted_lat(latencies);
    std::sort(sorted_lat.begin(), sorted_lat.end());
    double min_ms = sorted_lat.front();
    double max_ms = sorted_lat.back();
    double median_ms = sorted_lat[sorted_lat.size() / 2];
    double sum = 0.0;
    for (double v : sorted_lat) sum += v;
    double mean_ms = sum / static_cast<double>(sorted_lat.size());

    printf("{\n");
    printf("  \"config_slug\": \"%s\",\n", cfg->config_slug);

    /* output.result — flat list from first output buffer, bf16 -> float */
    printf("  \"output\": {\n    \"result\": [");
    if (!outputs.empty()) {
        const auto& buf = outputs[0];
        for (size_t i = 0; i < buf.size(); i++) {
            if (i > 0) printf(",");
            printf("%.6f", __bfloat162float(buf[i]));
        }
    }
    printf("],\n");
    printf("    \"metadata\": {\n");
    printf("      \"rank\": %d,\n", cfg->rank);
    printf("      \"shape_kind\": \"%s\",\n", cfg->shape_kind);
    printf("      \"input_size\": %d,\n", cfg->input_size);
    printf("      \"shape\": %s\n", cfg->shape_json);
    printf("    }\n  },\n");

    /* correctness stub — real correctness is computed by evaluate.py */
    printf("  \"correctness\": {\n");
    printf("    \"metadata\": {\n");
    printf("      \"rank\": %d,\n", cfg->rank);
    printf("      \"shape_kind\": \"%s\",\n", cfg->shape_kind);
    printf("      \"input_size\": %d,\n", cfg->input_size);
    printf("      \"shape\": %s\n", cfg->shape_json);
    printf("    },\n");
    printf("    \"passed\": true,\n");
    printf("    \"max_abs_error\": 0.0,\n");
    printf("    \"mean_abs_error\": 0.0\n");
    printf("  },\n");

    /* performance */
    printf("  \"performance\": {\n");
    printf("    \"metadata\": {\n");
    printf("      \"rank\": %d,\n", cfg->rank);
    printf("      \"shape_kind\": \"%s\",\n", cfg->shape_kind);
    printf("      \"input_size\": %d,\n", cfg->input_size);
    printf("      \"shape\": %s\n", cfg->shape_json);
    printf("    },\n");
    printf("    \"latency_ms\": {\n");
    printf("      \"min\": %.6f,\n", min_ms);
    printf("      \"median\": %.6f,\n", median_ms);
    printf("      \"max\": %.6f,\n", max_ms);
    printf("      \"mean\": %.6f\n", mean_ms);
    printf("    },\n");
    printf("    \"runs\": %d\n", static_cast<int>(latencies.size()));
    printf("  },\n");

    /* summary */
    printf("  \"summary\": {\n");
    printf("    \"metadata\": {\n");
    printf("      \"rank\": %d,\n", cfg->rank);
    printf("      \"shape_kind\": \"%s\",\n", cfg->shape_kind);
    printf("      \"input_size\": %d,\n", cfg->input_size);
    printf("      \"shape\": %s\n", cfg->shape_json);
    printf("    },\n");
    printf("    \"latency_ms\": {\n");
    printf("      \"min\": %.6f,\n", min_ms);
    printf("      \"median\": %.6f,\n", median_ms);
    printf("      \"max\": %.6f,\n", max_ms);
    printf("      \"mean\": %.6f\n", mean_ms);
    printf("    },\n");
    printf("    \"runs\": %d\n", static_cast<int>(latencies.size()));
    printf("  }\n");
    printf("}\n");
}

/* ------------------------------------------------------------------ */
/* main                                                               */
/* ------------------------------------------------------------------ */
int main() {
    HarnessConfig cfg;
    parse_config(&cfg);

    /* Buffer layout: convention-based.
     * Inputs:  num_inputs  buffers, each input_size BF16 elements.
     * Outputs: num_outputs buffers, each input_size BF16 elements.
     * Override counts via env vars if needed. */
    const int num_inputs  = env_int("CUDA_EXEC_PARAM_NUM_INPUTS",
                           env_int("CUDA_EXEC_HARNESS_NUM_INPUTS", 2));
    const int num_outputs = env_int("CUDA_EXEC_PARAM_NUM_OUTPUTS",
                           env_int("CUDA_EXEC_HARNESS_NUM_OUTPUTS", 1));
    const size_t elem_bytes = static_cast<size_t>(cfg.input_size) * sizeof(__nv_bfloat16);

    /* Allocate and fill host inputs (BF16 arange) */
    std::vector<std::vector<__nv_bfloat16>> h_inputs(num_inputs);
    for (int i = 0; i < num_inputs; i++) {
        h_inputs[i].resize(cfg.input_size);
        fill_arange(h_inputs[i].data(), cfg.input_size);
    }

    /* Allocate device inputs */
    std::vector<__nv_bfloat16*> d_inputs(num_inputs, nullptr);
    for (int i = 0; i < num_inputs; i++) {
        if (cudaMalloc(&d_inputs[i], elem_bytes) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc input %d failed\n", i);
            return 2;
        }
        cudaMemcpy(d_inputs[i], h_inputs[i].data(), elem_bytes, cudaMemcpyHostToDevice);
    }

    /* Allocate device outputs */
    std::vector<__nv_bfloat16*> d_outputs(num_outputs, nullptr);
    for (int i = 0; i < num_outputs; i++) {
        if (cudaMalloc(&d_outputs[i], elem_bytes) != cudaSuccess) {
            fprintf(stderr, "cudaMalloc output %d failed\n", i);
            return 3;
        }
        cudaMemset(d_outputs[i], 0, elem_bytes);
    }

    /* Create stream */
    cudaStream_t stream;
    cudaStreamCreate(&stream);

    /* L2 cache flush buffer (Triton do_bench / NVBench pattern) */
    int l2_attr = 0;
    cudaDeviceGetAttribute(&l2_attr, cudaDevAttrL2CacheSize, 0);
    size_t l2_size = static_cast<size_t>(l2_attr > 0 ? l2_attr : 0);
    void* l2_flush_buf = nullptr;
    if (l2_size > 0) {
        cudaMalloc(&l2_flush_buf, l2_size);
    }

    /* Warmup */
    for (int i = 0; i < cfg.num_warmups; i++) {
        int rc = kernel_run(d_inputs.data(), num_inputs,
                            d_outputs.data(), num_outputs,
                            cfg.input_size, stream);
        cudaStreamSynchronize(stream);
        if (rc != 0) {
            fprintf(stderr, "kernel_run warmup failed: rc=%d\n", rc);
            return 4;
        }
    }

    /* Timed trials with CUDA events */
    std::vector<double> latencies;
    for (int i = 0; i < cfg.num_trials; i++) {
        cudaEvent_t start_ev, end_ev;
        cudaEventCreate(&start_ev);
        cudaEventCreate(&end_ev);

        /* Flush L2 cache before each trial */
        if (l2_flush_buf != nullptr) {
            cudaMemsetAsync(l2_flush_buf, 0, l2_size, stream);
        }

        cudaEventRecord(start_ev, stream);
        int rc = kernel_run(d_inputs.data(), num_inputs,
                            d_outputs.data(), num_outputs,
                            cfg.input_size, stream);
        cudaEventRecord(end_ev, stream);
        cudaEventSynchronize(end_ev);

        if (rc != 0) {
            fprintf(stderr, "kernel_run trial %d failed: rc=%d\n", i, rc);
            return 5;
        }

        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_ev, end_ev);
        latencies.push_back(static_cast<double>(ms));

        cudaEventDestroy(start_ev);
        cudaEventDestroy(end_ev);
    }

    /* Copy outputs to host (BF16) */
    std::vector<std::vector<__nv_bfloat16>> h_outputs(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        h_outputs[i].resize(cfg.input_size);
        cudaMemcpy(h_outputs[i].data(), d_outputs[i],
                   elem_bytes, cudaMemcpyDeviceToHost);
    }

    /* Print structured JSON */
    print_json(&cfg, latencies, h_outputs);

    /* Cleanup */
    if (l2_flush_buf) cudaFree(l2_flush_buf);
    cudaStreamDestroy(stream);
    for (auto* p : d_inputs)  cudaFree(p);
    for (auto* p : d_outputs) cudaFree(p);

    return 0;
}
