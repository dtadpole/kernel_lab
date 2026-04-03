# FA4 Benchmark — 2026-04-03 (Unified Kernel + Real CuTe DSL)

## Hardware

| Property | Value |
|----------|-------|
| GPU | NVIDIA H100 SXM5 |
| Architecture | SM 9.0a (Hopper) |
| SMs | 132 |
| Peak BF16 Tensor Core | 989.5 TFLOPS (FP32 accum, dense) |
| DRAM Bandwidth | 3,352 GB/s (HBM3) |
| Driver | 550.90.07 |
| CUDA Toolkit | 12.8 |

## Benchmark Configuration

- **Host**: devvm8490, GPU index 4
- **Measurement**: CUDA event timing, L2 flush + fresh random inputs per trial, 10 warmup + 20 timed trials, median reported
- **Correctness**: Bit-exact match vs warp-specialized baseline (67M elements, both causal and non-causal)
- **All configs**: H=16, D=128, total_tokens=B×S=32768

## Implementations

1. **cuDNN** (`cudnn.py`): `torch.nn.functional.scaled_dot_product_attention` with `CUDNN_ATTENTION` backend forced. Venv: `~/.cuda_exec_service/.venv`
2. **FA4 CuTe DSL** (`cutedsl.py`): `flash_attn.cute.flash_attn_func` — SM90 WGMMA+TMA kernel via CuTe DSL JIT. Venv: `~/.fa4_venv` (cuda-bindings==12.8.0, `CUTE_DSL_ARCH=sm_90a`)
3. **Generated CUDA** (`generated.cu`): Unified architecture — 128 threads (4 warps), Q cached in registers, cooperative K/V loading, single-buffered K/V with SMEM reuse. Compiled with `nvcc -arch=sm_90a -O3`.

## Performance

| Config | cuDNN (TFLOPS) | CuTe DSL (TFLOPS) | Generated (TFLOPS) | DSL vs cuDNN | Gen vs cuDNN |
|--------|---------------|-------------------|--------------------|--------------|--------------| 
| causal b8-s4096 | 497.1 | 571.9 | 250.7 | 1.15x | 0.50x |
| causal b4-s8192 | 544.2 | 659.8 | 270.6 | 1.21x | 0.50x |
| causal b2-s16384 | 548.0 | 710.3 | 280.7 | 1.30x | 0.51x |
| noncausal b8-s4096 | 601.6 | 711.6 | 278.5 | 1.18x | 0.46x |
| noncausal b4-s8192 | 583.8 | 762.9 | 293.6 | 1.31x | 0.50x |
| noncausal b2-s16384 | 564.2 | 739.1 | 298.4 | 1.31x | 0.53x |

## Generated Kernel Resource Usage

| Metric | Value |
|--------|-------|
| Threads/block | 128 (4 warps) |
| Registers/thread | 224 |
| Spill stores | 0 |
| Spill loads | 0 |
| SMEM/block | 32 KB |
| Barriers | 1 (__syncthreads) |
| Occupancy | 2 blocks/SM |

## Analysis

- **FA4 CuTe DSL** is the fastest: 572–763 TFLOPS (1.15–1.31x vs cuDNN)
- **cuDNN**: 497–602 TFLOPS (vendor-optimized baseline)
- **Generated CUDA**: 251–298 TFLOPS (0.46–0.53x cuDNN, ~50% of vendor performance)
- Generated kernel achieves ~27% of H100 peak BF16 (989.5 TFLOPS)

### Remaining gap (Generated → CuTe DSL/cuDNN)

The ~2x gap is due to synchronous `mma.sync.m16n8k16`:
- Each MMA stalls the warp for ~4 cycles while tensor core executes
- Softmax scalar work (230 FPU + 38 MUFU instructions) cannot overlap with MMA
- 136 LDSM (ldmatrix) instructions load operands from SMEM to registers per iteration

FA4 CuTe DSL and cuDNN use SM90-native **WGMMA** (async warp group MMA) which:
- Executes MMA asynchronously — warp continues softmax while tensor core runs
- Uses shared memory descriptors — eliminates LDSM overhead
- Has 2x higher throughput per clock (m64nNk16 vs m16n8k16)
