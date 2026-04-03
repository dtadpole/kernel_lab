/*
 * CUTLASS 3.x WGMMA-based BF16 matrix multiplication for SM90 (H100 Hopper).
 *
 * Uses the CUTLASS CollectiveBuilder to construct an optimized kernel with:
 *   - TMA (Tensor Memory Accelerator) for global→shared memory loads
 *   - WGMMA (Warpgroup MMA) for compute — warpgroup-level m64nNk16 operations
 *   - Warp-specialized cooperative scheduling
 *   - Persistent tile scheduler for work distribution
 *   - Automatic stage count selection based on SMEM capacity
 *
 * Two tile configurations dispatched by matrix size:
 *   Big:   128×256×64 tile — for M,N ≥ 2048 (high arithmetic intensity)
 *   Small: 128×128×64 tile — for M,N < 2048
 *
 * kernel_run contract:
 *   extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
 *                             __nv_bfloat16** outputs, int num_outputs,
 *                             int n, cudaStream_t stream);
 */

#include <cuda_bf16.h>
#include <cuda.h>
#include <cstdio>
#include <cstdlib>

#include "cute/tensor.hpp"
#include "cutlass/cutlass.h"
#include "cutlass/bfloat16.h"
#include "cutlass/gemm/dispatch_policy.hpp"
#include "cutlass/gemm/collective/collective_builder.hpp"
#include "cutlass/epilogue/collective/collective_builder.hpp"
#include "cutlass/gemm/device/gemm_universal_adapter.h"
#include "cutlass/gemm/kernel/gemm_universal.hpp"
#include "cutlass/gemm/kernel/tile_scheduler.hpp"
#include "cutlass/util/packed_stride.hpp"

using namespace cute;

/* =========================================================================
 * GEMM type definitions using CollectiveBuilder
 * ========================================================================= */

using ElementA = cutlass::bfloat16_t;
using ElementB = cutlass::bfloat16_t;
using ElementC = cutlass::bfloat16_t;
using ElementD = cutlass::bfloat16_t;
using ElementAccumulator = float;
using ElementCompute = float;
using ElementScalar = float;

// A is row-major (M×K), B is column-major (K×N from PyTorch's B.t())
using LayoutA = cutlass::layout::RowMajor;
using LayoutB = cutlass::layout::ColumnMajor;
using LayoutC = cutlass::layout::RowMajor;
using LayoutD = cutlass::layout::RowMajor;

static constexpr int AlignmentA = 16 / sizeof(ElementA);  // 8
static constexpr int AlignmentB = 16 / sizeof(ElementB);  // 8
static constexpr int AlignmentC = 16 / sizeof(ElementC);  // 8
static constexpr int AlignmentD = 16 / sizeof(ElementD);  // 8

// Epilogue: linear combination D = alpha * Acc + beta * C (we use alpha=1, beta=0)
using EpilogueOp = cutlass::epilogue::fusion::LinearCombination<
    ElementD, ElementCompute, ElementC, ElementScalar>;

// ---------- Big kernel: 128×256×64 for large matrices ----------

using TileShape_Big = Shape<_128, _256, _64>;
using ClusterShape_Big = Shape<_1, _1, _1>;

using CollectiveEpilogue_Big = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_Big, ClusterShape_Big,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop_Big = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_Big, ClusterShape_Big,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_Big::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel_Big = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop_Big,
    CollectiveEpilogue_Big,
    cutlass::gemm::PersistentScheduler
>;
using Gemm_Big = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_Big>;

// ---------- Small kernel: 128×128×64 for smaller matrices ----------

using TileShape_Small = Shape<_128, _128, _64>;
using ClusterShape_Small = Shape<_1, _1, _1>;

using CollectiveEpilogue_Small = typename cutlass::epilogue::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    TileShape_Small, ClusterShape_Small,
    cutlass::epilogue::collective::EpilogueTileAuto,
    ElementAccumulator, ElementCompute,
    ElementC, LayoutC, AlignmentC,
    ElementD, LayoutD, AlignmentD,
    cutlass::epilogue::TmaWarpSpecializedCooperative,
    EpilogueOp
>::CollectiveOp;

using CollectiveMainloop_Small = typename cutlass::gemm::collective::CollectiveBuilder<
    cutlass::arch::Sm90, cutlass::arch::OpClassTensorOp,
    ElementA, LayoutA, AlignmentA,
    ElementB, LayoutB, AlignmentB,
    ElementAccumulator,
    TileShape_Small, ClusterShape_Small,
    cutlass::gemm::collective::StageCountAutoCarveout<
        static_cast<int>(sizeof(typename CollectiveEpilogue_Small::SharedStorage))>,
    cutlass::gemm::KernelTmaWarpSpecializedCooperative
>::CollectiveOp;

using GemmKernel_Small = cutlass::gemm::kernel::GemmUniversal<
    Shape<int, int, int, int>,
    CollectiveMainloop_Small,
    CollectiveEpilogue_Small,
    cutlass::gemm::PersistentScheduler
>;
using Gemm_Small = cutlass::gemm::device::GemmUniversalAdapter<GemmKernel_Small>;

/* =========================================================================
 * GEMM runner — handles argument setup and kernel launch
 * ========================================================================= */

template <typename GemmType>
static int run_gemm(int M, int N, int K,
                    const __nv_bfloat16* A, const __nv_bfloat16* B, __nv_bfloat16* C,
                    cudaStream_t stream) {
    using StrideA = typename GemmType::GemmKernel::StrideA;
    using StrideB = typename GemmType::GemmKernel::StrideB;
    using StrideC = typename GemmType::GemmKernel::StrideC;
    using StrideD = typename GemmType::GemmKernel::StrideD;

    // Row-major A (M×K): stride = (K, 1, 0)
    StrideA stride_A = cutlass::make_cute_packed_stride(StrideA{}, cute::make_shape(M, K, 1));
    // Column-major B: stride computed from (N, K)
    StrideB stride_B = cutlass::make_cute_packed_stride(StrideB{}, cute::make_shape(N, K, 1));
    // Row-major C/D (M×N)
    StrideC stride_C = cutlass::make_cute_packed_stride(StrideC{}, cute::make_shape(M, N, 1));
    StrideD stride_D = cutlass::make_cute_packed_stride(StrideD{}, cute::make_shape(M, N, 1));

    // Hardware info for persistent scheduler
    cutlass::KernelHardwareInfo hw_info;
    hw_info.device_id = 0;
    cudaGetDevice(&hw_info.device_id);
    hw_info.sm_count = cutlass::KernelHardwareInfo::query_device_multiprocessor_count(hw_info.device_id);

    typename GemmType::Arguments arguments{
        cutlass::gemm::GemmUniversalMode::kGemm,
        {M, N, K, 1},  // problem shape (M, N, K, batch=1)
        {
            reinterpret_cast<const ElementA*>(A), stride_A,
            reinterpret_cast<const ElementB*>(B), stride_B
        },
        {
            {1.0f, 0.0f}, // alpha=1, beta=0
            reinterpret_cast<const ElementC*>(C), stride_C,
            reinterpret_cast<ElementD*>(C), stride_D
        },
        hw_info
    };

    GemmType gemm;
    size_t workspace_size = GemmType::get_workspace_size(arguments);
    void* workspace = nullptr;
    if (workspace_size > 0) {
        cudaMalloc(&workspace, workspace_size);
    }

    auto status = gemm.can_implement(arguments);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS can_implement failed: %d\n", (int)status);
        if (workspace) cudaFree(workspace);
        return -1;
    }

    status = gemm.initialize(arguments, workspace, stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS initialize failed: %d\n", (int)status);
        if (workspace) cudaFree(workspace);
        return -1;
    }

    status = gemm.run(stream);
    if (status != cutlass::Status::kSuccess) {
        fprintf(stderr, "CUTLASS run failed: %d\n", (int)status);
        if (workspace) cudaFree(workspace);
        return -1;
    }

    if (workspace) cudaFree(workspace);
    return 0;
}

/* =========================================================================
 * kernel_run — eval harness contract
 * ========================================================================= */

extern "C" int kernel_run(__nv_bfloat16** inputs, int num_inputs,
                          __nv_bfloat16** outputs, int num_outputs,
                          int n, cudaStream_t stream) {
    if (num_inputs < 2 || num_outputs < 1) return -1;

    const __nv_bfloat16* A = inputs[0];
    const __nv_bfloat16* B = inputs[1];
    __nv_bfloat16* C = outputs[0];

    // n is input_size (total elements per tensor). For square matrices: dim = sqrt(n).
    int dim = 1;
    while (dim * dim < n) dim++;
    if (dim * dim != n) {
        fprintf(stderr, "input_size %d is not a perfect square\n", n);
        return -1;
    }
    int M = dim, N = dim, K = dim;

    if (M >= 2048 && N >= 2048) {
        return run_gemm<Gemm_Big>(M, N, K, A, B, C, stream);
    } else {
        return run_gemm<Gemm_Small>(M, N, K, A, B, C, stream);
    }
}
