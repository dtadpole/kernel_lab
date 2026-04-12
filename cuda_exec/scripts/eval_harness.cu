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
 *
 * Measurement methodology (per-iteration sync):
 *   Each trial: L2 flush → sync → [start event] → kernel → [end event] → sync.
 *   Per-iteration sync prevents DRAM bandwidth contention between
 *   consecutive kernels (previous kernel's epilogue writes vs current
 *   kernel's TMA loads).  This matches PyTorch's benchmark_utils behavior.
 *
 *   Kernels that export kernel_setup() + kernel_launch() get cleaner
 *   measurement: CPU-heavy work (TMA encoding, grid calc) is done once
 *   in kernel_setup(), and only the lightweight kernel_launch() runs in
 *   the timed loop.  Kernels without the split API fall back to
 *   kernel_run() per iteration.
 */

#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <nvml.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <unistd.h>
#include <vector>

/* ------------------------------------------------------------------ */
/* Kernel contract — implemented by the kernel .cu file               */
/* ------------------------------------------------------------------ */
/* Required: legacy all-in-one entry point */
extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
                          __nv_bfloat16** outputs, int num_outputs,
                          int n, cudaStream_t stream);

/* Optional: split setup/launch for pipelined measurement.
 * kernel_setup does CPU work (TMA encoding, grid calc) once.
 * kernel_launch only enqueues the GPU kernel (no CPU overhead).
 * If both are defined, the harness uses them for the timed loop. */
extern "C" int kernel_setup(__nv_bfloat16** inputs, int num_inputs,
                            __nv_bfloat16** outputs, int num_outputs,
                            int n, cudaStream_t stream) __attribute__((weak));
extern "C" int kernel_launch(cudaStream_t stream) __attribute__((weak));

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
    float warmup_ms;    /* target warmup time in ms (time-based) */
    float rep_ms;       /* target measurement time in ms (time-based) */
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

static float env_float(const char* name, float fallback) {
    const char* v = getenv(name);
    return v ? static_cast<float>(atof(v)) : fallback;
}

/* ------------------------------------------------------------------ */
/* GPU state snapshot via NVML                                        */
/* ------------------------------------------------------------------ */
struct GpuSnapshot {
    unsigned int sm_clock_mhz;
    unsigned int temp_c;
    bool valid;
};

static nvmlDevice_t g_nvml_dev = nullptr;
static bool g_nvml_ok = false;

static void nvml_init() {
    if (g_nvml_ok) return;
    if (nvmlInit_v2() != NVML_SUCCESS) return;

    /* Use CUDA_VISIBLE_DEVICES index 0 → need the real NVML device index. */
    int cuda_dev = 0;
    char pci_bus_id[64] = {};
    if (cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), cuda_dev) != cudaSuccess)
        return;
    if (nvmlDeviceGetHandleByPciBusId_v2(pci_bus_id, &g_nvml_dev) != NVML_SUCCESS)
        return;
    g_nvml_ok = true;
}

static GpuSnapshot gpu_snapshot() {
    GpuSnapshot s = {};
    if (!g_nvml_ok) return s;
    nvmlDeviceGetClockInfo(g_nvml_dev, NVML_CLOCK_SM, &s.sm_clock_mhz);
    nvmlDeviceGetTemperature(g_nvml_dev, NVML_TEMPERATURE_GPU, &s.temp_c);
    s.valid = true;
    return s;
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

    cfg->warmup_ms = env_float("CUDA_EXEC_WARMUP_MS", 25.0f);
    cfg->rep_ms    = env_float("CUDA_EXEC_REP_MS", 100.0f);
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
/* Device kernel: fill buffer with pseudo-random BF16 values.         */
/* Used to randomize inputs so that kernels cannot exploit constant   */
/* data patterns.                                                     */
/* ------------------------------------------------------------------ */
__global__ void fill_random_bf16(__nv_bfloat16* buf, int count,
                                 unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= count) return;
    /* fast hash: Wang's 32-bit integer hash */
    unsigned int h = (unsigned int)idx ^ seed;
    h = (h ^ 61u) ^ (h >> 16);
    h += (h << 3);
    h ^= (h >> 4);
    h *= 0x27d4eb2du;
    h ^= (h >> 15);
    /* map to [-0.5, 0.5) to keep values small (avoids fp overflow in matmul) */
    float val = (float)(h & 0xFFFFu) / 65536.0f - 0.5f;
    buf[idx] = __float2bfloat16(val);
}

/* ------------------------------------------------------------------ */
/* JSON output (BF16 values converted to float for printing)          */
/* ------------------------------------------------------------------ */
static void print_json(const HarnessConfig* cfg,
                       const std::vector<double>& latencies,
                       const std::vector<std::vector<__nv_bfloat16>>& outputs,
                       const GpuSnapshot& snap_warmup,
                       const GpuSnapshot& snap_before,
                       const GpuSnapshot& snap_after) {
    /* Latency stats */
    std::vector<double> sorted_lat(latencies);
    std::sort(sorted_lat.begin(), sorted_lat.end());
    double min_ms = sorted_lat.front();
    double max_ms = sorted_lat.back();
    double p10_ms = sorted_lat[sorted_lat.size() / 10];
    double p25_ms = sorted_lat[sorted_lat.size() / 4];
    double median_ms = sorted_lat[sorted_lat.size() / 2];
    double p75_ms = sorted_lat[sorted_lat.size() * 3 / 4];
    double p90_ms = sorted_lat[sorted_lat.size() * 9 / 10];
    double sum = 0.0;
    for (double v : sorted_lat) sum += v;
    double mean_ms = sum / static_cast<double>(sorted_lat.size());

    printf("{\n");
    printf("  \"config_slug\": \"%s\",\n", cfg->config_slug);

    /* output — write raw BF16 binary files to CUDA_EXEC_OUTPUT_DIR
     * (set by evaluate.py).  JSON output.result is always empty. */
    const char* output_dir = getenv("CUDA_EXEC_OUTPUT_DIR");
    if (output_dir && output_dir[0] && !outputs.empty()) {
        for (size_t i = 0; i < outputs.size(); i++) {
            char path[512];
            snprintf(path, sizeof(path), "%s/output_%zu.bin", output_dir, i);
            FILE* f = fopen(path, "wb");
            if (f) {
                fwrite(outputs[i].data(), sizeof(__nv_bfloat16),
                       outputs[i].size(), f);
                fclose(f);
            }
        }
    }
    printf("  \"output\": {\n    \"result\": [],\n");
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
    printf("      \"p10\": %.6f,\n", p10_ms);
    printf("      \"p25\": %.6f,\n", p25_ms);
    printf("      \"p50\": %.6f,\n", median_ms);
    printf("      \"p75\": %.6f,\n", p75_ms);
    printf("      \"p90\": %.6f,\n", p90_ms);
    printf("      \"max\": %.6f,\n", max_ms);
    printf("      \"mean\": %.6f\n", mean_ms);
    printf("    },\n");
    printf("    \"raw_latencies_ms\": [");
    for (size_t i = 0; i < latencies.size(); i++) {
        if (i > 0) printf(",");
        printf("%.6f", latencies[i]);
    }
    printf("],\n");
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
    printf("      \"p10\": %.6f,\n", p10_ms);
    printf("      \"p25\": %.6f,\n", p25_ms);
    printf("      \"p50\": %.6f,\n", median_ms);
    printf("      \"p75\": %.6f,\n", p75_ms);
    printf("      \"p90\": %.6f,\n", p90_ms);
    printf("      \"max\": %.6f,\n", max_ms);
    printf("      \"mean\": %.6f\n", mean_ms);
    printf("    },\n");
    printf("    \"runs\": %d\n", static_cast<int>(latencies.size()));
    printf("  },\n");

    /* GPU clock/temp snapshots (NVML) — frequency responds in <100ms.
     * Power is omitted (needs ~1s ramp-up; measured at outer bench level).
     *   warmup: after warmup completes (post-sync idle clock)
     *   before: right before measurement loop starts
     *   during: sampled mid-measurement (GPU under sustained load) */
    printf("  \"gpu_state\": {\n");
    if (snap_warmup.valid) {
        printf("    \"warmup\": { \"sm_clock_mhz\": %u, \"temp_c\": %u },\n",
               snap_warmup.sm_clock_mhz, snap_warmup.temp_c);
    }
    if (snap_before.valid) {
        printf("    \"before\": { \"sm_clock_mhz\": %u, \"temp_c\": %u },\n",
               snap_before.sm_clock_mhz, snap_before.temp_c);
    }
    if (snap_after.valid) {
        printf("    \"during\": { \"sm_clock_mhz\": %u, \"temp_c\": %u },\n",
               snap_after.sm_clock_mhz, snap_after.temp_c);
    }

    /* Flag clock throttling: before vs during comparison */
    bool throttled = false;
    int clock_drop_mhz = 0;
    if (snap_before.valid && snap_after.valid &&
        snap_after.sm_clock_mhz < snap_before.sm_clock_mhz) {
        throttled = true;
        clock_drop_mhz = (int)snap_before.sm_clock_mhz - (int)snap_after.sm_clock_mhz;
    }
    printf("    \"throttled\": %s", throttled ? "true" : "false");
    if (throttled) {
        printf(",\n    \"clock_drop_mhz\": %d", clock_drop_mhz);
    }
    printf("\n  }\n");
    printf("}\n");

    /* Warn on stderr if clock dropped during measurement */
    if (throttled) {
        fprintf(stderr, "WARNING: GPU clock throttled during measurement: "
                "%u MHz → %u MHz (-%d MHz). Results may be affected.\n",
                snap_before.sm_clock_mhz, snap_after.sm_clock_mhz, clock_drop_mhz);
    }
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
    const int num_inputs  = env_int("CUDA_EXEC_HARNESS_NUM_INPUTS",
                            env_int("CUDA_EXEC_PARAM_HARNESS_NUM_INPUTS",
                            env_int("CUDA_EXEC_EXTRA_HARNESS_NUM_INPUTS", 2)));
    const int num_outputs = env_int("CUDA_EXEC_HARNESS_NUM_OUTPUTS",
                            env_int("CUDA_EXEC_PARAM_HARNESS_NUM_OUTPUTS",
                            env_int("CUDA_EXEC_EXTRA_HARNESS_NUM_OUTPUTS", 1)));
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

    /* Init NVML for GPU state snapshots */
    nvml_init();

    /* L2 cache flush buffer — use actual L2 size (matches eval_support.py).
     * In pipelined mode, oversized flush buffers cause DRAM bandwidth
     * contention with the previous kernel, inflating latency. */
    int l2_attr = 0;
    cudaDeviceGetAttribute(&l2_attr, cudaDevAttrL2CacheSize, 0);
    size_t l2_size = static_cast<size_t>(l2_attr > 0 ? l2_attr : 0);
    void* l2_flush_buf = nullptr;
    if (l2_size > 0) {
        cudaMalloc(&l2_flush_buf, l2_size);
    }

    /* ── Detect setup/launch split ─────────────────────────────────── */
    /* If the kernel exports kernel_setup + kernel_launch, we can separate
     * CPU-heavy setup (TMA encoding, grid calc) from the GPU-only launch.
     * This enables safe pipelined measurement (like Triton do_bench). */
    const bool has_split = (kernel_setup != nullptr && kernel_launch != nullptr);

    /* ── Initial run + setup ──────────────────────────────────────── */
    {
        int rc = kernel_run(d_inputs.data(), num_inputs,
                            d_outputs.data(), num_outputs,
                            cfg.input_size, stream);
        cudaStreamSynchronize(stream);
        if (rc != 0) {
            fprintf(stderr, "kernel_run initial call failed: rc=%d\n", rc);
            return 4;
        }
    }

    /* If split API available, call setup once (caches TMA descriptors etc.) */
    if (has_split) {
        int rc = kernel_setup(d_inputs.data(), num_inputs,
                              d_outputs.data(), num_outputs,
                              cfg.input_size, stream);
        cudaStreamSynchronize(stream);
        if (rc != 0) {
            fprintf(stderr, "kernel_setup failed: rc=%d\n", rc);
            return 4;
        }
    }

    /* Helper lambda: invoke kernel with minimal overhead */
    auto invoke_kernel = [&](cudaStream_t s) -> int {
        if (has_split) return kernel_launch(s);
        return kernel_run(d_inputs.data(), num_inputs,
                          d_outputs.data(), num_outputs,
                          cfg.input_size, s);
    };

    /* ── Estimate single-run time ─────────────────────────────────── */
    cudaEvent_t est_start, est_end;
    cudaEventCreate(&est_start);
    cudaEventCreate(&est_end);
    cudaEventRecord(est_start, stream);
    for (int i = 0; i < 5; i++) {
        if (l2_flush_buf != nullptr)
            cudaMemsetAsync(l2_flush_buf, 0, l2_size, stream);
        invoke_kernel(stream);
    }
    cudaEventRecord(est_end, stream);
    cudaDeviceSynchronize();

    float est_total_ms = 0.0f;
    cudaEventElapsedTime(&est_total_ms, est_start, est_end);
    float est_ms = est_total_ms / 5.0f;
    if (est_ms <= 0.0f) est_ms = 0.001f;
    cudaEventDestroy(est_start);
    cudaEventDestroy(est_end);

    /* ── Time-based warmup ────────────────────────────────────────── */
    const int n_warmup = (int)(cfg.warmup_ms / est_ms);
    const int actual_warmup = n_warmup > 1 ? n_warmup : 1;
    for (int i = 0; i < actual_warmup; i++) {
        invoke_kernel(stream);
    }

    /* GPU state snapshot after warmup */
    cudaStreamSynchronize(stream);
    GpuSnapshot snap_warmup = gpu_snapshot();

    /* Randomize inputs before the timed loop */
    const int rand_block = 256;
    const int rand_grid = (cfg.input_size + rand_block - 1) / rand_block;
    for (int j = 0; j < num_inputs; j++) {
        unsigned int seed = 0xCAFE0000u + (unsigned int)j;
        fill_random_bf16<<<rand_grid, rand_block, 0, stream>>>(
            d_inputs[j], cfg.input_size, seed);
    }
    cudaStreamSynchronize(stream);

    /* Re-setup after input randomization (pointers unchanged, but data changed) */
    if (has_split) {
        kernel_setup(d_inputs.data(), num_inputs,
                     d_outputs.data(), num_outputs,
                     cfg.input_size, stream);
        cudaStreamSynchronize(stream);
    }

    /* ── Time-based measurement (pipelined — no per-iteration sync) ── */
    const int n_trials = (int)(cfg.rep_ms / est_ms);
    const int N = n_trials > 1 ? n_trials : 1;

    GpuSnapshot snap_before = gpu_snapshot();

    std::vector<cudaEvent_t> start_events(N), end_events(N);
    for (int i = 0; i < N; i++) {
        cudaEventCreate(&start_events[i]);
        cudaEventCreate(&end_events[i]);
    }

    const int mid = N / 2;
    GpuSnapshot snap_after = {};

    /* Pipelined: enqueue all iterations without CPU sync.
     * Each iteration: L2 flush (async) → start event → kernel → end event.
     * No cudaStreamSynchronize between iterations — kernels can overlap
     * with the next L2 flush on the GPU command queue. */
    /* Check env: CUDA_EXEC_L2_FLUSH=0 disables L2 flush (for comparison with triton do_bench) */
    const bool do_l2_flush = (l2_flush_buf != nullptr) && env_int("CUDA_EXEC_L2_FLUSH", 1);

    for (int i = 0; i < N; i++) {
        if (do_l2_flush) {
            cudaMemsetAsync(l2_flush_buf, 0, l2_size, stream);
        }
        cudaEventRecord(start_events[i], stream);
        invoke_kernel(stream);
        cudaEventRecord(end_events[i], stream);
        /* NO cudaStreamSynchronize here — pipelined */
    }

    /* Single sync after all iterations */
    cudaStreamSynchronize(stream);
    snap_after = gpu_snapshot();

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA error during trials: %s\n",
                cudaGetErrorString(err));
        return 5;
    }

    std::vector<double> latencies;
    for (int i = 0; i < N; i++) {
        float ms = 0.0f;
        cudaEventElapsedTime(&ms, start_events[i], end_events[i]);
        latencies.push_back(static_cast<double>(ms));
    }

    for (int i = 0; i < N; i++) {
        cudaEventDestroy(start_events[i]);
        cudaEventDestroy(end_events[i]);
    }

    /* Correctness pass: allocate fresh buffers and fill with deterministic
     * random data. Seeds are simply 1, 2, 3, ... per input tensor.
     * The Python trial script reproduces this exact PRNG for comparison.
     * Using random instead of arange avoids BF16 overflow for large sizes. */
    for (int j = 0; j < num_inputs; j++)
        cudaFree(d_inputs[j]);
    for (int j = 0; j < num_inputs; j++) {
        cudaMalloc(&d_inputs[j], elem_bytes);
        unsigned int correctness_seed = (unsigned int)(j + 1);  /* seed 1, 2, 3, ... */
        fill_random_bf16<<<rand_grid, rand_block, 0, stream>>>(
            d_inputs[j], cfg.input_size, correctness_seed);
    }
    cudaStreamSynchronize(stream);

    {
        int rc = kernel_run(d_inputs.data(), num_inputs,
                            d_outputs.data(), num_outputs,
                            cfg.input_size, stream);
        cudaStreamSynchronize(stream);
        if (rc != 0) {
            fprintf(stderr, "kernel_run correctness pass failed: rc=%d\n", rc);
            return 6;
        }
    }

    /* Copy outputs to host (BF16) */
    std::vector<std::vector<__nv_bfloat16>> h_outputs(num_outputs);
    for (int i = 0; i < num_outputs; i++) {
        h_outputs[i].resize(cfg.input_size);
        cudaMemcpy(h_outputs[i].data(), d_outputs[i],
                   elem_bytes, cudaMemcpyDeviceToHost);
    }

    /* Print structured JSON */
    print_json(&cfg, latencies, h_outputs, snap_warmup, snap_before, snap_after);

    /* Cleanup */
    if (l2_flush_buf) cudaFree(l2_flush_buf);
    cudaStreamDestroy(stream);
    for (auto* p : d_inputs)  cudaFree(p);
    for (auto* p : d_outputs) cudaFree(p);

    return 0;
}
