/*
 * Sample BF16 matmul for SM100 — TCGEN05 + TMA (v7).
 *
 * Uses Blackwell TCGEN05 MMA with TMEM accumulators and TMA 128B swizzle.
 * K=16 per MMA call (kind::f16, cta_group::1). 4 K-steps per TILE_K=64.
 *
 * SMEM descriptor: LBO=0, SBO=1024, swizzle mode 2 (128B).
 * idesc: transpose_b=1.
 * K-stepping: add to full 64-bit descriptor (safe since LBO=0).
 *
 * Architecture: 128 threads (4 warps), cooperative.
 *   Thread 0 issues TMA loads + TCGEN05 MMA (single-thread semantics).
 *   All threads participate in TMEM zeroing and epilogue.
 * Tile: 128×64×64. Persistent CTAs.
 *
 * Tunable: STAGES (pipeline stages for TMA).
 */
#include <cuda_bf16.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cmath>

#ifndef STAGES
#define STAGES 1
#endif

#define TILE_M      128
#define TILE_N      64
#define TILE_K      64
#define MMA_K       16
#define K_STEPS     (TILE_K / MMA_K)
#define NUM_WARPS   4
#define THREADS     (NUM_WARPS * 32)

#define A_BYTES     (TILE_M * TILE_K * 2)   /* 16384 */
#define B_BYTES     (TILE_K * TILE_N * 2)   /* 8192 */
#define STAGE_BYTES (A_BYTES + B_BYTES)      /* 24576 */

#define SMEM_DATA   (STAGES * STAGE_BYTES)
#define MBAR_OFF    ((SMEM_DATA + 255) & ~255)
#define TMEM_OFF    (MBAR_OFF + 256)
#define SMEM_TOTAL  (TMEM_OFF + 64)

/* idesc: 128×64 BF16→F32 dense, transpose_b=1 */
#define IDESC_128x64 ((1u<<4)|(1u<<7)|(1u<<10)|(1u<<16)|(8u<<17)|(8u<<24))

/* ── TCGEN05 + mbarrier + TMA PTX ───────────────────────────── */
__device__ __forceinline__ void _alloc(uint32_t*d,uint32_t n){unsigned a=__cvta_generic_to_shared(d);asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"::"r"(a),"r"(n):"memory");}
__device__ __forceinline__ void _dealloc(uint32_t t,uint32_t n){asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(t),"r"(n):"memory");}
__device__ __forceinline__ void _mma(uint32_t d,uint64_t a,uint64_t b,uint32_t id,bool en){asm volatile("{\n .reg .pred p;\n setp.ne.b32 p,%4,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}"::"r"(d),"l"(a),"l"(b),"r"(id),"r"((uint32_t)en):"memory");}
__device__ __forceinline__ void _commit(uint64_t*b){unsigned a=__cvta_generic_to_shared(b);asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(a):"memory");}
__device__ __forceinline__ void _fb(){asm volatile("tcgen05.fence::before_thread_sync;":::"memory");}
__device__ __forceinline__ void _fa(){asm volatile("tcgen05.fence::after_thread_sync;":::"memory");}
__device__ __forceinline__ void _wl(){asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");}
__device__ __forceinline__ void _ws(){asm volatile("tcgen05.wait::st.sync.aligned;":::"memory");}
__device__ __forceinline__ void _st1(uint32_t t,uint32_t v){asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],{%1};"::"r"(t),"r"(v):"memory");}
/* Load all 64 F32 accum values from TMEM in one shot (32x32b.x64) */
__device__ __forceinline__ void _ld64(uint32_t(&o)[64],uint32_t t){
    asm volatile(
        "tcgen05.ld.sync.aligned.32x32b.x64.b32 "
        "{%0,%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,"
        "%16,%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,"
        "%32,%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,"
        "%48,%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63}, [%64];"
        :"=r"(o[0]),"=r"(o[1]),"=r"(o[2]),"=r"(o[3]),"=r"(o[4]),"=r"(o[5]),"=r"(o[6]),"=r"(o[7]),
         "=r"(o[8]),"=r"(o[9]),"=r"(o[10]),"=r"(o[11]),"=r"(o[12]),"=r"(o[13]),"=r"(o[14]),"=r"(o[15]),
         "=r"(o[16]),"=r"(o[17]),"=r"(o[18]),"=r"(o[19]),"=r"(o[20]),"=r"(o[21]),"=r"(o[22]),"=r"(o[23]),
         "=r"(o[24]),"=r"(o[25]),"=r"(o[26]),"=r"(o[27]),"=r"(o[28]),"=r"(o[29]),"=r"(o[30]),"=r"(o[31]),
         "=r"(o[32]),"=r"(o[33]),"=r"(o[34]),"=r"(o[35]),"=r"(o[36]),"=r"(o[37]),"=r"(o[38]),"=r"(o[39]),
         "=r"(o[40]),"=r"(o[41]),"=r"(o[42]),"=r"(o[43]),"=r"(o[44]),"=r"(o[45]),"=r"(o[46]),"=r"(o[47]),
         "=r"(o[48]),"=r"(o[49]),"=r"(o[50]),"=r"(o[51]),"=r"(o[52]),"=r"(o[53]),"=r"(o[54]),"=r"(o[55]),
         "=r"(o[56]),"=r"(o[57]),"=r"(o[58]),"=r"(o[59]),"=r"(o[60]),"=r"(o[61]),"=r"(o[62]),"=r"(o[63])
        :"r"(t):"memory");}
__device__ __forceinline__ void _mi(uint64_t*m,unsigned c){unsigned a=__cvta_generic_to_shared(m);asm volatile("mbarrier.init.shared.b64 [%0],%1;\n"::"r"(a),"r"(c));}
__device__ __forceinline__ void _mw(uint64_t*m,unsigned p){unsigned a=__cvta_generic_to_shared(m);unsigned r;do{asm volatile("{\n .reg .pred q;\n mbarrier.try_wait.parity.shared.b64 q,[%1],%2;\n selp.u32 %0,1,0,q;\n}\n":"=r"(r):"r"(a),"r"(p));}while(!r);}
__device__ __forceinline__ void _ma_tx(uint64_t*m,unsigned tx){unsigned a=__cvta_generic_to_shared(m);asm volatile("mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;\n"::"r"(a),"r"(tx));}
__device__ __forceinline__ void tma2d(void*dst,const void*desc,int c0,int c1,uint64_t*bar){unsigned s=__cvta_generic_to_shared(dst),b=__cvta_generic_to_shared(bar);asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];\n"::"r"(s),"l"(desc),"r"(c0),"r"(c1),"r"(b):"memory");}

/* SMEM descriptor: LBO=0, SBO from param, 128B swizzle */
__device__ __forceinline__
uint64_t _desc(const void*p, int sbo_bytes){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    uint64_t d=0;
    d|=(uint64_t)((a>>4)&0x3FFF);
    d|=(uint64_t)(((sbo_bytes>>4)&0x3FFF))<<32;
    d|=(uint64_t)(1)<<46;
    d|=(uint64_t)(2)<<61;
    return d;}

/* TMEM → (row,col) for 32x32b layout (from Triton DistributedLinearLayout):
 *   reg_bases  = [[0,1],[0,2],[0,4],[0,8],[0,16],[0,32]]  → reg → col
 *   lane_bases = [[1,0],[2,0],[4,0],[8,0],[16,0]]          → lane → row
 *   warp_bases = [[32,0],[64,0]]                            → warp → row
 * Simple: col = reg_idx, row = lane ^ (warp * 32) */
__device__ __forceinline__
void tmem_coords(int warp, int lane, int reg_idx, int& row, int& col) {
    col = reg_idx;  /* reg bits directly map to column */
    row = lane;     /* lane bits directly map to row within 32 */
    row ^= (warp & 1) * 32;
    row ^= ((warp >> 1) & 1) * 64;
}

/* ── Kernel ──────────────────────────────────────────────────── */
__global__ void __launch_bounds__(THREADS, 1)
matmul_tcgen05(
    const __nv_bfloat16* __restrict__ A,
    const __nv_bfloat16* __restrict__ B,
    __nv_bfloat16* __restrict__ C,
    int M, int N, int K,
    const __grid_constant__ CUtensorMap tma_A,
    const __grid_constant__ CUtensorMap tma_B)
{
    extern __shared__ char smem[];
    uint64_t* load_bar = (uint64_t*)(smem + MBAR_OFF);
    uint64_t* mma_bar  = (uint64_t*)(smem + MBAR_OFF + 64);
    uint32_t* tmem_slot = (uint32_t*)(smem + TMEM_OFF);

    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lid = tid % 32;
    const int numK = (K + TILE_K - 1) / TILE_K;

    const int grid_m = (M + TILE_M - 1) / TILE_M;
    const int grid_n = (N + TILE_N - 1) / TILE_N;
    const int total_tiles = grid_m * grid_n;

    /* Init */
    if (wid == 0) _alloc(tmem_slot, TILE_N);
    __syncthreads();
    uint32_t tmem_base = *tmem_slot;

    if (tid == 0) {
        _mi(load_bar, 1);
        _mi(mma_bar, 1);
        asm volatile("prefetch.tensormap [%0];\n"::"l"(&tma_A):"memory");
        asm volatile("prefetch.tensormap [%0];\n"::"l"(&tma_B):"memory");
    }
    __syncthreads();

    /* Persistent tile loop */
    for (int tile = blockIdx.x; tile < total_tiles; tile += gridDim.x) {
        int ctaRow = (tile / grid_n) * TILE_M;
        int ctaCol = (tile % grid_n) * TILE_N;

        /* Zero TMEM for this tile */
        for (int c = lid; c < TILE_N; c += 32) _st1(tmem_base + c, 0);
        _ws();
        __syncthreads();

        /* K-loop — simple: init barrier, TMA, wait, MMA, repeat */
        for (int kt = 0; kt < numK; kt++) {
            if (tid == 0) {
                /* Init + TMA load */
                _mi(load_bar, 1);
                _ma_tx(load_bar, A_BYTES + B_BYTES);
                tma2d(smem, &tma_A, kt * TILE_K, ctaRow, load_bar);
                tma2d(smem + A_BYTES, &tma_B, ctaCol, kt * TILE_K, load_bar);
            }
            /* Wait for TMA — always parity 0 since we reinit each time */
            _mw(load_bar, 0);
            __syncthreads();

            /* MMA: 4 K-steps */
            if (tid == 0) {
                _mi(mma_bar, 1);
                uint64_t ad0 = _desc(smem, 8 * TILE_K * 2);
                uint64_t bd0 = _desc(smem + A_BYTES, 8 * TILE_N * 2);
                for (int kk = 0; kk < K_STEPS; kk++) {
                    uint64_t ad = ad0 + (uint64_t)((kk * MMA_K * 2) >> 4);
                    uint64_t bd = bd0 + (uint64_t)((kk * MMA_K * TILE_N * 2) >> 4);
                    _mma(tmem_base, ad, bd, IDESC_128x64, (kt > 0 || kk > 0));
                }
                _commit(mma_bar);
            }
            _mw(mma_bar, 0);
            __syncthreads();
        }

        /* Epilogue: TMEM → registers → bf16 → global */
        _fb(); __syncthreads(); _fa();

        /* Load all 64 accum values from TMEM in one shot */
        uint32_t regs[64];
        _ld64(regs, tmem_base);
        _wl();

        /* Write to global: reg→col, lane→row (32x32b layout) */
        for (int r = 0; r < 64; r++) {
            int row, col;
            tmem_coords(wid, lid, r, row, col);
            int gRow = ctaRow + row;
            int gCol = ctaCol + col;
            if (gRow < M && gCol < N)
                C[(size_t)gRow * N + gCol] = __float2bfloat16(__int_as_float(regs[r]));
        }
        __syncthreads();
    }

    /* Cleanup */
    __syncthreads();
    if (wid == 0) _dealloc(tmem_base, TILE_N);
}

/* ── Host ────────────────────────────────────────────────────── */
typedef CUresult (*enc_fn)(CUtensorMap*,CUtensorMapDataType,cuuint32_t,void*,
    const cuuint64_t*,const cuuint64_t*,const cuuint32_t*,const cuuint32_t*,
    CUtensorMapInterleave,CUtensorMapSwizzle,CUtensorMapL2promotion,CUtensorMapFloatOOBfill);
static enc_fn s_enc = nullptr;
static bool init_enc() {
    if(s_enc)return true;
    void*h=dlopen("libcuda.so.1",RTLD_LAZY|RTLD_NOLOAD);
    if(!h)h=dlopen("libcuda.so.1",RTLD_LAZY);if(!h)return false;
    typedef CUresult(*gp)(const char*,void**,int,cuuint64_t,CUdriverProcAddressQueryResult*);
    gp g=(gp)dlsym(h,"cuGetProcAddress_v2");if(!g)g=(gp)dlsym(h,"cuGetProcAddress");
    CUdriverProcAddressQueryResult st;
    return g("cuTensorMapEncodeTiled",(void**)&s_enc,12000,
             CU_GET_PROC_ADDRESS_DEFAULT,&st)==CUDA_SUCCESS&&s_enc;
}

extern "C" int kernel_run(
    __nv_bfloat16** inputs, int num_inputs,
    __nv_bfloat16** outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    int dim=(int)sqrtf((float)n);
    if(dim*dim!=n)return 1;
    if(!init_enc())return -1;

    const __nv_bfloat16*A=inputs[0],*B=inputs[1];
    __nv_bfloat16*C_=outputs[0];
    int M=dim,N=dim,K=dim;

    CUtensorMap tA,tB;
    {cuuint64_t d[2]={(cuuint64_t)K,(cuuint64_t)M};cuuint64_t s[1]={(cuuint64_t)(K*2)};
     cuuint32_t b[2]={(cuuint32_t)TILE_K,(cuuint32_t)TILE_M};cuuint32_t e[2]={1,1};
     if(s_enc(&tA,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)A,d,s,b,e,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))return -2;}
    {cuuint64_t d[2]={(cuuint64_t)N,(cuuint64_t)K};cuuint64_t s[1]={(cuuint64_t)(N*2)};
     cuuint32_t b[2]={(cuuint32_t)TILE_N,(cuuint32_t)TILE_K};cuuint32_t e[2]={1,1};
     if(s_enc(&tB,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)B,d,s,b,e,
        CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
        CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
        CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE))return -3;}

    int sms=0;cudaDeviceGetAttribute(&sms,cudaDevAttrMultiProcessorCount,0);
    if(sms<=0)sms=148;
    int gm=(M+TILE_M-1)/TILE_M, gn=(N+TILE_N-1)/TILE_N;
    int tiles=gm*gn, blocks=min(tiles,sms);

    cudaFuncSetAttribute(matmul_tcgen05,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_TOTAL);
    matmul_tcgen05<<<blocks,THREADS,SMEM_TOTAL,stream>>>(A,B,C_,M,N,K,tA,tB);
    return 0;
}
