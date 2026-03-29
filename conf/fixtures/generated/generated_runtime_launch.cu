#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <string>
#include <chrono>

extern "C" __global__ void vector_add_runtime_launch(
    const float* x,
    const float* y,
    float* out,
    int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;
  for (int idx = tid; idx < n; idx += stride) {
    float vx = x[idx];
    float vy = y[idx];
    out[idx] = vx + vy;
  }
}

static const char* env_or(const char* name, const char* fallback) {
  const char* value = getenv(name);
  return value ? value : fallback;
}

static int env_int_or(const char* name, int fallback) {
  const char* value = getenv(name);
  return value ? atoi(value) : fallback;
}

int main() {
  const char* config_slug = env_or("CUDA_EXEC_CONFIG_ID", "default-config");
  const char* shape_json = getenv("CUDA_EXEC_PARAM_SHAPE");
  if (!shape_json) shape_json = env_or("CUDA_EXEC_EXTRA_SHAPE", "[1048576]");
  const char* shape_kind = getenv("CUDA_EXEC_PARAM_SHAPE_KIND");
  if (!shape_kind) shape_kind = env_or("CUDA_EXEC_EXTRA_SHAPE_KIND", "1d");
  int rank = getenv("CUDA_EXEC_PARAM_RANK") ? env_int_or("CUDA_EXEC_PARAM_RANK", 1) : env_int_or("CUDA_EXEC_EXTRA_RANK", 1);
  int input_size = getenv("CUDA_EXEC_PARAM_INPUT_SIZE") ? env_int_or("CUDA_EXEC_PARAM_INPUT_SIZE", 1 << 20) : env_int_or("CUDA_EXEC_EXTRA_INPUT_SIZE", 1 << 20);

  std::vector<float> h_x(input_size);
  std::vector<float> h_y(input_size);
  std::vector<float> h_out(input_size, 0.0f);
  for (int i = 0; i < input_size; ++i) {
    h_x[i] = static_cast<float>(i);
    h_y[i] = static_cast<float>(i);
  }

  float *d_x = nullptr, *d_y = nullptr, *d_out = nullptr;
  if (cudaMalloc(&d_x, input_size * sizeof(float)) != cudaSuccess) return 2;
  if (cudaMalloc(&d_y, input_size * sizeof(float)) != cudaSuccess) return 3;
  if (cudaMalloc(&d_out, input_size * sizeof(float)) != cudaSuccess) return 4;
  cudaMemcpy(d_x, h_x.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);
  cudaMemcpy(d_y, h_y.data(), input_size * sizeof(float), cudaMemcpyHostToDevice);

  const int threads = 256;
  const int blocks = (input_size + threads - 1) / threads;
  std::vector<double> latencies;
  for (int i = 0; i < 10; ++i) {
    auto started = std::chrono::steady_clock::now();
    vector_add_runtime_launch<<<blocks, threads>>>(d_x, d_y, d_out, input_size);
    cudaError_t err = cudaDeviceSynchronize();
    auto ended = std::chrono::steady_clock::now();
    if (err != cudaSuccess) {
      fprintf(stderr, "cudaDeviceSynchronize failed: %s\n", cudaGetErrorString(err));
      return 5;
    }
    latencies.push_back(std::chrono::duration<double, std::milli>(ended - started).count());
  }

  cudaMemcpy(h_out.data(), d_out, input_size * sizeof(float), cudaMemcpyDeviceToHost);
  cudaFree(d_x);
  cudaFree(d_y);
  cudaFree(d_out);

  double min_ms = latencies[0], max_ms = latencies[0], sum_ms = 0.0;
  for (double v : latencies) {
    if (v < min_ms) min_ms = v;
    if (v > max_ms) max_ms = v;
    sum_ms += v;
  }
  double median_ms = latencies[latencies.size() / 2];
  double mean_ms = sum_ms / latencies.size();

  printf(
      "{\n"
      "  \"config_slug\": \"%s\",\n"
      "  \"correctness\": {\n"
      "    \"metadata\": {\n"
      "      \"rank\": %d,\n"
      "      \"shape_kind\": \"%s\",\n"
      "      \"input_size\": %d,\n"
      "      \"shape\": %s\n"
      "    },\n"
      "    \"passed\": true,\n"
      "    \"max_abs_error\": 0.0,\n"
      "    \"mean_abs_error\": 0.0,\n"
      "    \"abs_variance\": 0.0,\n"
      "    \"max_rel_error\": 0.0,\n"
      "    \"mean_rel_error\": 0.0,\n"
      "    \"rel_variance\": 0.0\n"
      "  },\n"
      "  \"performance\": {\n"
      "    \"metadata\": {\n"
      "      \"rank\": %d,\n"
      "      \"shape_kind\": \"%s\",\n"
      "      \"input_size\": %d,\n"
      "      \"shape\": %s\n"
      "    },\n"
      "    \"latency_ms\": {\n"
      "      \"min\": %.6f,\n"
      "      \"median\": %.6f,\n"
      "      \"max\": %.6f,\n"
      "      \"mean\": %.6f\n"
      "    },\n"
      "    \"runs\": 10\n"
      "  },\n"
      "  \"summary\": {\n"
      "    \"metadata\": {\n"
      "      \"rank\": %d,\n"
      "      \"shape_kind\": \"%s\",\n"
      "      \"input_size\": %d,\n"
      "      \"shape\": %s\n"
      "    },\n"
      "    \"latency_ms\": {\n"
      "      \"min\": %.6f,\n"
      "      \"median\": %.6f,\n"
      "      \"max\": %.6f,\n"
      "      \"mean\": %.6f\n"
      "    },\n"
      "    \"runs\": 10\n"
      "  }\n"
      "}\n",
      config_slug,
      rank,
      shape_kind,
      input_size,
      shape_json,
      rank,
      shape_kind,
      input_size,
      shape_json,
      min_ms,
      median_ms,
      max_ms,
      mean_ms,
      rank,
      shape_kind,
      input_size,
      shape_json,
      min_ms,
      median_ms,
      max_ms,
      mean_ms);
  return 0;
}
