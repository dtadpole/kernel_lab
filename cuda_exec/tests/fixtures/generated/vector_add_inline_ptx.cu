#include <cuda_runtime.h>
#include <stdint.h>
#include <stdio.h>

extern "C" __global__ void vector_add_inline_ptx(
    const float* x,
    const float* y,
    float* out,
    int n) {
  int tid = blockIdx.x * blockDim.x + threadIdx.x;
  int stride = blockDim.x * gridDim.x;

  for (int idx = tid; idx < n; idx += stride) {
    float vx = 0.0f;
    float vy = 0.0f;
    float vz = 0.0f;

    asm volatile("ld.global.f32 %0, [%1];" : "=f"(vx) : "l"(x + idx));
    asm volatile("ld.global.f32 %0, [%1];" : "=f"(vy) : "l"(y + idx));
    asm volatile("add.rn.f32 %0, %1, %2;" : "=f"(vz) : "f"(vx), "f"(vy));
    asm volatile("st.global.f32 [%0], %1;" :: "l"(out + idx), "f"(vz));
  }
}

int main() {
  constexpr int n = 1 << 20;
  printf("vector_add_inline_ptx fixture reached main, n=%d\n", n);
  return 0;
}
