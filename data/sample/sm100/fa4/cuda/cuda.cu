/*
 * Flash Attention forward — BF16, SM100 TCGEN05 sample.
 *
 * Simplified FA4 for B200 using Blackwell-native TCGEN05 instructions.
 * Non-causal only. BLOCK_Q=128, BLOCK_KV=128, DIM=128.
 *
 * Algorithm:
 *   For each Q-block (128 tokens):
 *     O_acc = 0, rowmax = -inf, rowsum = 0
 *     For each KV-block:
 *       S = Q × K^T  (128×128, via TCGEN05 MMA with D=128 reduction)
 *       P = row_softmax(S, update rowmax/rowsum, rescale O_acc)
 *       O_acc += P × V  (128×128 accumulation)
 *     O = O_acc / rowsum
 *
 * Memory layout: Q,K,V,O each (B, S, H, D) BF16, contiguous.
 *
 * Architecture: 128 threads (4 warps), cooperative.
 *   Thread 0 issues TMA loads + TCGEN05 MMA (single-thread semantics).
 *   All threads participate in softmax + TMEM readback.
 *   Two TMEM allocations: S_tmem (QK result) and O_tmem (PV accumulator).
 *
 * Uses TMA + 128B swizzle. SMEM descriptors: LBO=0, SBO=1024, swizzle mode 2.
 */
#include <cuda_bf16.h>
#include <cuda.h>
#include <dlfcn.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <cfloat>

/* ── Tile parameters ─────────────────────────────────────────── */
#define BLOCK_Q     128
#define BLOCK_KV    128
#define DIM         128
#define MMA_K       16
#define NUM_WARPS   4
#define THREADS     (NUM_WARPS * 32)

/* SMEM: Q[128×128] + K[128×128] + V[128×128] = 3 × 32768 = 98304 */
#define TILE_BYTES  (128 * 128 * 2)   /* 32768 bytes per 128×128 bf16 tile */
/* Half-DIM tiles: 128×64 bf16 = 16384 bytes each */
#define HALF_BYTES  (128 * 64 * 2)    /* 16384 */
/* We load Q once, K and V per KV-block. Use half-DIM TMA (box=64×128). */
#define Q_OFF       0
#define K_OFF       (2 * HALF_BYTES)  /* after Q_lo + Q_hi */
#define V_OFF       (4 * HALF_BYTES)  /* after K_lo + K_hi */
#define DATA_BYTES  (6 * HALF_BYTES)  /* 98304 */
#define MBAR_OFF    ((DATA_BYTES + 255) & ~255)
#define TMEM_OFF    (MBAR_OFF + 256)
#define SMEM_TOTAL  (TMEM_OFF + 128)

/* idesc: 128×128 BF16→F32, transpose_b=1 */
#define IDESC_128x128 ((1u<<4)|(1u<<7)|(1u<<10)|(1u<<16)|(16u<<17)|(8u<<24))
/* idesc: 128×128 BF16→F32, transpose_b=1 — same for PV GEMM */
#define IDESC_PV IDESC_128x128

/* ── TCGEN05 + TMA PTX (reuse from matmul sample) ────────────── */
__device__ __forceinline__ void _alloc(uint32_t*d,uint32_t n){unsigned a=__cvta_generic_to_shared(d);asm volatile("tcgen05.alloc.cta_group::1.sync.aligned.shared::cta.b32 [%0],%1;"::"r"(a),"r"(n):"memory");}
__device__ __forceinline__ void _dealloc(uint32_t t,uint32_t n){asm volatile("tcgen05.dealloc.cta_group::1.sync.aligned.b32 %0,%1;"::"r"(t),"r"(n):"memory");}
__device__ __forceinline__ void _mma(uint32_t d,uint64_t a,uint64_t b,uint32_t id,bool en){asm volatile("{\n .reg .pred p;\n setp.ne.b32 p,%4,0;\n tcgen05.mma.cta_group::1.kind::f16 [%0],%1,%2,%3,p;\n}"::"r"(d),"l"(a),"l"(b),"r"(id),"r"((uint32_t)en):"memory");}
__device__ __forceinline__ void _commit(uint64_t*b){unsigned a=__cvta_generic_to_shared(b);asm volatile("tcgen05.commit.cta_group::1.mbarrier::arrive::one.b64 [%0];"::"r"(a):"memory");}
__device__ __forceinline__ void _fb(){asm volatile("tcgen05.fence::before_thread_sync;":::"memory");}
__device__ __forceinline__ void _fa(){asm volatile("tcgen05.fence::after_thread_sync;":::"memory");}
__device__ __forceinline__ void _wl(){asm volatile("tcgen05.wait::ld.sync.aligned;":::"memory");}
__device__ __forceinline__ void _ws(){asm volatile("tcgen05.wait::st.sync.aligned;":::"memory");}
__device__ __forceinline__ void _st1(uint32_t t,uint32_t v){asm volatile("tcgen05.st.sync.aligned.32x32b.x1.b32 [%0],{%1};"::"r"(t),"r"(v):"memory");}

/* Load 64 F32 values from TMEM (covers 64 columns for one warp-set) */
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

/* Store 64 F32 values to TMEM */
__device__ __forceinline__ void _st64(uint32_t t,const uint32_t(&v)[64]){
    asm volatile(
        "tcgen05.st.sync.aligned.32x32b.x64.b32 [%0], "
        "{%1,%2,%3,%4,%5,%6,%7,%8,%9,%10,%11,%12,%13,%14,%15,%16,"
        "%17,%18,%19,%20,%21,%22,%23,%24,%25,%26,%27,%28,%29,%30,%31,%32,"
        "%33,%34,%35,%36,%37,%38,%39,%40,%41,%42,%43,%44,%45,%46,%47,%48,"
        "%49,%50,%51,%52,%53,%54,%55,%56,%57,%58,%59,%60,%61,%62,%63,%64};"
        ::"r"(t),
         "r"(v[0]),"r"(v[1]),"r"(v[2]),"r"(v[3]),"r"(v[4]),"r"(v[5]),"r"(v[6]),"r"(v[7]),
         "r"(v[8]),"r"(v[9]),"r"(v[10]),"r"(v[11]),"r"(v[12]),"r"(v[13]),"r"(v[14]),"r"(v[15]),
         "r"(v[16]),"r"(v[17]),"r"(v[18]),"r"(v[19]),"r"(v[20]),"r"(v[21]),"r"(v[22]),"r"(v[23]),
         "r"(v[24]),"r"(v[25]),"r"(v[26]),"r"(v[27]),"r"(v[28]),"r"(v[29]),"r"(v[30]),"r"(v[31]),
         "r"(v[32]),"r"(v[33]),"r"(v[34]),"r"(v[35]),"r"(v[36]),"r"(v[37]),"r"(v[38]),"r"(v[39]),
         "r"(v[40]),"r"(v[41]),"r"(v[42]),"r"(v[43]),"r"(v[44]),"r"(v[45]),"r"(v[46]),"r"(v[47]),
         "r"(v[48]),"r"(v[49]),"r"(v[50]),"r"(v[51]),"r"(v[52]),"r"(v[53]),"r"(v[54]),"r"(v[55]),
         "r"(v[56]),"r"(v[57]),"r"(v[58]),"r"(v[59]),"r"(v[60]),"r"(v[61]),"r"(v[62]),"r"(v[63])
        :"memory");}

__device__ __forceinline__ void _mi(uint64_t*m,unsigned c){unsigned a=__cvta_generic_to_shared(m);asm volatile("mbarrier.init.shared.b64 [%0],%1;\n"::"r"(a),"r"(c));}
__device__ __forceinline__ void _mw(uint64_t*m,unsigned p){unsigned a=__cvta_generic_to_shared(m);unsigned r;do{asm volatile("{\n .reg .pred q;\n mbarrier.try_wait.parity.shared.b64 q,[%1],%2;\n selp.u32 %0,1,0,q;\n}\n":"=r"(r):"r"(a),"r"(p));}while(!r);}
__device__ __forceinline__ void _ma_tx(uint64_t*m,unsigned tx){unsigned a=__cvta_generic_to_shared(m);asm volatile("mbarrier.arrive.expect_tx.shared.b64 _,[%0],%1;\n"::"r"(a),"r"(tx));}
__device__ __forceinline__ void tma2d(void*dst,const void*desc,int c0,int c1,uint64_t*bar){unsigned s=__cvta_generic_to_shared(dst),b=__cvta_generic_to_shared(bar);asm volatile("cp.async.bulk.tensor.2d.shared::cta.global.tile.mbarrier::complete_tx::bytes [%0],[%1,{%2,%3}],[%4];\n"::"r"(s),"l"(desc),"r"(c0),"r"(c1),"r"(b):"memory");}

/* SMEM descriptor: LBO=0, 128B swizzle */
__device__ __forceinline__
uint64_t _desc(const void*p, int sbo_bytes){
    uint32_t a=(uint32_t)__cvta_generic_to_shared(p);
    uint64_t d=0;
    d|=(uint64_t)((a>>4)&0x3FFF);
    d|=(uint64_t)(((sbo_bytes>>4)&0x3FFF))<<32;
    d|=(uint64_t)(1)<<46;
    d|=(uint64_t)(2)<<61;
    return d;}

/* Fast math */
__device__ __forceinline__ float fast_exp2f(float x){float r;asm("ex2.approx.ftz.f32 %0,%1;":"=f"(r):"f"(x));return r;}
__device__ __forceinline__ float fast_rcp(float x){float r;asm("rcp.approx.ftz.f32 %0,%1;":"=f"(r):"f"(x));return r;}

__device__ __host__ constexpr int cdiv(int a,int b){return (a+b-1)/b;}

/* ── TMEM coords for 128×128 (32x32b layout) ───────────────── */
/* reg_bases=[[0,1],[0,2],[0,4],[0,8],[0,16],[0,32],[0,64]]
 * lane_bases=[[1,0],[2,0],[4,0],[8,0],[16,0]]
 * warp_bases=[[32,0],[64,0]]
 * → col = reg_idx, row = lane ^ (warp_offset) */
__device__ __forceinline__
void tmem_coords_128(int warp, int lane, int reg_idx, int& row, int& col) {
    col = reg_idx;  /* 0..127 */
    row = lane;
    row ^= (warp & 1) * 32;
    row ^= ((warp >> 1) & 1) * 64;
}

/* ── Kernel ──────────────────────────────────────────────────── */
__global__ void __launch_bounds__(THREADS, 1)
fa_tcgen05(
    __nv_bfloat16* __restrict__ O_global,
    int B, int S, int H, int len_q, int len_kv, bool is_causal,
    const __grid_constant__ CUtensorMap tma_Q,
    const __grid_constant__ CUtensorMap tma_K,
    const __grid_constant__ CUtensorMap tma_V,
    const __grid_constant__ CUtensorMap tma_O)
{
    extern __shared__ char smem[];
    uint64_t* load_bar = (uint64_t*)(smem + MBAR_OFF);
    uint64_t* mma_bar  = (uint64_t*)(smem + MBAR_OFF + 64);
    uint32_t* tmem_slot = (uint32_t*)(smem + TMEM_OFF);

    const int tid = threadIdx.x;
    const int wid = tid / 32;
    const int lid = tid % 32;

    const int bid = blockIdx.x;
    const int num_q_blocks = cdiv(len_q, BLOCK_Q);
    const int bs_id = bid / num_q_blocks;
    const int q_block_id = bid % num_q_blocks;
    const int batch_id = bs_id / H;
    const int head_id = bs_id % H;
    const int coord_head = head_id * DIM;
    const int batch_seq = batch_id * S;
    const int max_kv_iter = is_causal
        ? min(cdiv(len_kv, BLOCK_KV), cdiv((q_block_id + 1) * BLOCK_Q, BLOCK_KV))
        : cdiv(len_kv, BLOCK_KV);
    const float softmax_scale = rsqrtf((float)DIM);
    const float softmax_scale_log2 = softmax_scale * 1.4426950408889634f;

    /* Alloc TMEM: S (128 cols for QK scores 128×128), O_lo/O_hi (64 each) */
    if (wid == 0) {
        _alloc(tmem_slot, 128);     /* S_tmem — QK scores */
        _alloc(tmem_slot + 1, 64);  /* O_lo — output cols 0-63 */
        _alloc(tmem_slot + 2, 64);  /* O_hi — output cols 64-127 */
    }
    __syncthreads();
    uint32_t S_tmem = tmem_slot[0];
    uint32_t O_lo   = tmem_slot[1];
    uint32_t O_hi   = tmem_slot[2];

    if (tid == 0) {
        _mi(load_bar, 1);
        _mi(mma_bar, 1);
        asm volatile("prefetch.tensormap [%0];\n"::"l"(&tma_Q):"memory");
        asm volatile("prefetch.tensormap [%0];\n"::"l"(&tma_K):"memory");
        asm volatile("prefetch.tensormap [%0];\n"::"l"(&tma_V):"memory");
    }
    __syncthreads();

    /* Zero O accumulators */
    for (int c = lid; c < 64; c += 32) { _st1(O_lo + c, 0); _st1(O_hi + c, 0); }
    _ws();
    __syncthreads();

    /* Load Q (once per CTA) — two half-DIM TMA loads */
    if (tid == 0) {
        _mi(load_bar, 1);
        _ma_tx(load_bar, 2 * HALF_BYTES);
        tma2d(smem + Q_OFF,             &tma_Q, coord_head,          batch_seq + q_block_id * BLOCK_Q, load_bar);
        tma2d(smem + Q_OFF + HALF_BYTES, &tma_Q, coord_head + DIM/2, batch_seq + q_block_id * BLOCK_Q, load_bar);
    }
    _mw(load_bar, 0);
    __syncthreads();

    /* Per-thread online softmax state (from TMEM layout: each thread owns specific rows) */
    float rowmax = -FLT_MAX;
    float rowsum = 0.0f;

    /* KV-loop */
    for (int kv_id = 0; kv_id < max_kv_iter; kv_id++) {
        /* Load K and V for this KV block */
        if (tid == 0) {
            _mi(load_bar, 1);
            _ma_tx(load_bar, 2 * HALF_BYTES + 2 * HALF_BYTES);  /* K + V */
            tma2d(smem + K_OFF,             &tma_K, coord_head,          batch_seq + kv_id * BLOCK_KV, load_bar);
            tma2d(smem + K_OFF + HALF_BYTES, &tma_K, coord_head + DIM/2, batch_seq + kv_id * BLOCK_KV, load_bar);
            tma2d(smem + V_OFF,             &tma_V, coord_head,          batch_seq + kv_id * BLOCK_KV, load_bar);
            tma2d(smem + V_OFF + HALF_BYTES, &tma_V, coord_head + DIM/2, batch_seq + kv_id * BLOCK_KV, load_bar);
        }
        _mw(load_bar, 0);
        __syncthreads();

        /* ---- QK GEMM: S = Q × K^T ---- */
        /* Zero S_tmem (128 cols) */
        for (int c = lid; c < 128; c += 32) _st1(S_tmem + c, 0);
        _ws(); __syncthreads();

        if (tid == 0) {
            _mi(mma_bar, 1);

            /* Q is at smem+Q_OFF (128×128 bf16, stored as two 128×64 halves).
             * K is at smem+K_OFF (128×128 bf16, stored as two 128×64 halves).
             * For QK GEMM: C[128×128] = Q[128×D] × K^T[D×128]
             * D=128 → 8 MMA K-steps of K=16.
             * Q and K are each stored as [lo, hi] with 64 columns each. */

            /* Q half-lo: smem+Q_OFF, stride=64*2=128B, SBO=8*128=1024 */
            uint64_t q_lo = _desc(smem + Q_OFF, 8 * 64 * 2);
            uint64_t q_hi = _desc(smem + Q_OFF + HALF_BYTES, 8 * 64 * 2);
            /* K half-lo: smem+K_OFF */
            uint64_t k_lo = _desc(smem + K_OFF, 8 * 64 * 2);
            uint64_t k_hi = _desc(smem + K_OFF + HALF_BYTES, 8 * 64 * 2);

            /* 8 K-steps: first 4 from Q_lo×K_lo, next 4 from Q_hi×K_hi */
            for (int kk = 0; kk < 8; kk++) {
                uint64_t qd = (kk < 4) ? q_lo + (uint64_t)((kk * MMA_K * 2) >> 4)
                                       : q_hi + (uint64_t)(((kk-4) * MMA_K * 2) >> 4);
                uint64_t kd = (kk < 4) ? k_lo + (uint64_t)((kk * MMA_K * 2) >> 4)
                                       : k_hi + (uint64_t)(((kk-4) * MMA_K * 2) >> 4);
                _mma(S_tmem, qd, kd, IDESC_128x128, (kk > 0));
            }
            _commit(mma_bar);
        }
        _mw(mma_bar, 0);
        __syncthreads();

        /* ---- Read S from TMEM, apply softmax in registers ---- */
        _fb(); __syncthreads(); _fa();

        /* Read S (128 cols): two x64 loads at S_tmem and S_tmem+64.
         * S_tmem was allocated with 128 cols, so +64 addresses cols 64-127
         * for tcgen05.ld (NOT for tcgen05.mma — ld addressing is different). */
        uint32_t s_regs_lo[64], s_regs_hi[64];
        _ld64(s_regs_lo, S_tmem);
        _wl();
        _ld64(s_regs_hi, S_tmem + 64);
        _wl();

        /* Causal masking + softmax scaling */
        /* TMEM layout (32x32b): col = reg_idx, row = lane ^ warp_offset.
         * This thread's query position: q_block_id * BLOCK_Q + row. */
        int my_row = lid ^ ((wid & 1) * 32) ^ (((wid >> 1) & 1) * 64);
        int q_pos = q_block_id * BLOCK_Q + my_row;

        float local_max = -FLT_MAX;
        for (int i = 0; i < 64; i++) {
            float v = __int_as_float(s_regs_lo[i]) * softmax_scale;
            /* Causal mask: col i = kv position kv_id*BLOCK_KV + i */
            if (is_causal && (kv_id * BLOCK_KV + i) > q_pos) v = -FLT_MAX;
            s_regs_lo[i] = __float_as_int(v);
            local_max = fmaxf(local_max, v);
        }
        for (int i = 0; i < 64; i++) {
            float v = __int_as_float(s_regs_hi[i]) * softmax_scale;
            if (is_causal && (kv_id * BLOCK_KV + 64 + i) > q_pos) v = -FLT_MAX;
            s_regs_hi[i] = __float_as_int(v);
            local_max = fmaxf(local_max, v);
        }

        /* Online softmax: update global max and rescale */
        float new_max = fmaxf(rowmax, local_max);
        float rescale = (rowmax > -FLT_MAX) ? fast_exp2f((rowmax - new_max) * 1.4426950408889634f) : 0.0f;
        rowmax = new_max;

        /* Compute exp and sum */
        float local_sum = 0.0f;
        float neg_max = -new_max;
        for (int i = 0; i < 64; i++) {
            float v = fast_exp2f((__int_as_float(s_regs_lo[i]) + neg_max) * 1.4426950408889634f);
            s_regs_lo[i] = __float_as_int(v);
            local_sum += v;
        }
        for (int i = 0; i < 64; i++) {
            float v = fast_exp2f((__int_as_float(s_regs_hi[i]) + neg_max) * 1.4426950408889634f);
            s_regs_hi[i] = __float_as_int(v);
            local_sum += v;
        }

        /* Update running sum: rowsum = rowsum * rescale + local_sum */
        rowsum = rowsum * rescale + local_sum;

        /* ---- Rescale O by rescale factor ---- */
        /* Always execute _ld64/_st64 (they are .sync.aligned — all threads must participate).
         * Threads where rescale==1.0f effectively no-op. */
        {
            uint32_t o_vals[64];
            _ld64(o_vals, O_lo); _wl();
            for (int i = 0; i < 64; i++)
                o_vals[i] = __float_as_int(__int_as_float(o_vals[i]) * rescale);
            _st64(O_lo, o_vals); _ws();
            _ld64(o_vals, O_hi); _wl();
            for (int i = 0; i < 64; i++)
                o_vals[i] = __float_as_int(__int_as_float(o_vals[i]) * rescale);
            _st64(O_hi, o_vals); _ws();
        }
        __syncthreads();

        /* ---- PV GEMM: O += P × V via TCGEN05 MMA ---- */
        /* Write P (F32 in regs) to SMEM as BF16 with 128B swizzle layout.
         * The swizzle formula: byte_addr ^ (((byte_addr >> 7) & 7) << 4). */
        {
            int row, col;
            /* P is stored at K_OFF in SMEM (reuse K region, 128×128 bf16 = 32768 bytes).
             * We split into two halves: P_lo (col 0-63) at K_OFF, P_hi (col 64-127) at K_OFF+HALF_BYTES.
             * Each half is 128×64 bf16 = 16384 bytes with 128B swizzle. */
            for (int i = 0; i < 64; i++) {
                tmem_coords_128(wid, lid, i, row, col);
                /* col < 64 → P_lo half */
                int local_byte = (row * 64 + col) * 2;
                int swizzled = local_byte ^ (((local_byte >> 7) & 7) << 4);
                *(__nv_bfloat16*)(smem + K_OFF + swizzled) =
                    __float2bfloat16(__int_as_float(s_regs_lo[i]));
            }
            for (int i = 0; i < 64; i++) {
                tmem_coords_128(wid, lid, 64 + i, row, col);
                /* col >= 64 → P_hi half, local col = col - 64 */
                int local_col = col - 64;
                int local_byte = (row * 64 + local_col) * 2;
                int swizzled = local_byte ^ (((local_byte >> 7) & 7) << 4);
                *(__nv_bfloat16*)(smem + K_OFF + HALF_BYTES + swizzled) =
                    __float2bfloat16(__int_as_float(s_regs_hi[i]));
            }
        }
        __syncthreads();

        /* PV GEMM via TCGEN05: O[128×128] += P[128×128] × V[128×128]
         * P in SMEM at K_OFF (two 128×64 halves, swizzled).
         * V in SMEM at V_OFF (two 128×64 halves, TMA-swizzled).
         * O accumulates in O_tmem.
         *
         * K dimension = 128 (columns of P / rows of V).
         * MMA K=16, so 8 K-steps.
         * P_lo covers P columns 0-63, P_hi covers 64-127.
         * V_lo covers V columns 0-63 (output cols 0-63), V_hi covers 64-127.
         *
         * Actually, for PV: C[m][n] = sum_k P[m][k] * V[k][n]
         * P is MxK=128×128, V is KxN=128×128 (but V stored as rows×cols).
         * The K dimension of PV = 128 = BLOCK_KV.
         * P's "K" direction = P's column direction.
         * V's "K" direction = V's row direction.
         *
         * P is stored as two halves: P_lo[128×64] and P_hi[128×64].
         * Each half has stride 64 bf16 = 128 bytes per row.
         * V is stored as two halves: V_lo[128×64] and V_hi[128×64].
         * Each half has stride 64 bf16 = 128 bytes per row.
         *
         * For MMA K-stepping through P's columns (V's rows):
         * K-steps 0-3 use P_lo columns 0-63 and V rows 0-63.
         * K-steps 4-7 use P_hi columns 64-127 and V rows 64-127.
         *
         * But wait: V_lo covers V's first 64 COLUMNS (head_dim 0-63),
         * not first 64 ROWS. V is [K×DIM] = [128×128]. V_lo is V[0:128][0:64].
         * The K dimension of V is the ROW dimension (128 rows).
         *
         * For PV: we reduce over K=128, which means over V's ROWS.
         * V_lo[128×64] has ALL 128 K-rows but only 64 output columns.
         * So we need TWO PV GEMMs: one for output cols 0-63 (using V_lo),
         * one for output cols 64-127 (using V_hi).
         * Each PV GEMM: P[128×128] × V_half[128×64] → O_half[128×64].
         *
         * But 128×64 MMA → idesc N=64. And K=128 → 8 K-steps.
         * P has 128 columns stored as P_lo[128×64] and P_hi[128×64].
         * For K-stepping: K-steps 0-3 from P_lo, K-steps 4-7 from P_hi.
         * V_half has 128 rows × 64 cols, stride=64*2=128B. */

        /* PV GEMM via TCGEN05: O[128×128] += P[128×128] × V[128×128]
         * Use IDESC_128x128 (same as QK GEMM).
         * P is in SMEM at K_OFF as two contiguous 128×64 halves.
         * V is in SMEM at V_OFF as two contiguous 128×64 halves.
         * The 128-wide MMA reads across both halves (they're adjacent in SMEM).
         *
         * For PV: K dimension = BLOCK_KV = 128 (P's columns = V's rows).
         * P halves split K at 64: P_lo=K[0:64], P_hi=K[64:128].
         * V halves split N (DIM) at 64: V_lo=DIM[0:64], V_hi=DIM[64:128].
         *
         * K-stepping through P's columns:
         *   K-steps 0-3: advance within P_lo (K=0..63)
         *   K-steps 4-7: advance within P_hi (K=64..127)
         * V descriptor stays at V_lo base but MMA reads 128 V-columns (both halves).
         * V rows advance by MMA_K per K-step.
         *
         * BUT: V's K dimension is ROWS. V has 128 rows × 128 cols (two halves).
         * V_lo = V[0:128][0:64], V_hi = V[0:128][64:128].
         * Advancing V by K-step means advancing V's row index by MMA_K=16.
         * V_lo stride = 64*2 = 128B per row. K-step advance = 16*128 = 2048B.
         * But MMA needs 128 V-columns for N=128 output. V_lo only has 64 cols.
         * The MMA reads across V_lo into V_hi (they're contiguous). */
        /* PV GEMM: two passes for 128×128 output.
         * Pass 1: P[128×128] × V_lo[128×64] → O_tmem (cols 0-63)
         * Pass 2: P[128×128] × V_hi[128×64] → S_tmem (cols 0-63, used as O cols 64-127)
         * idesc 128×64 for each pass. K=128, 8 K-steps per pass. */
        {
            uint32_t idesc_pv = (1u<<4)|(1u<<7)|(1u<<10)|(1u<<16)|(8u<<17)|(8u<<24);
            uint64_t p_lo = _desc(smem + K_OFF, 8 * 64 * 2);
            uint64_t p_hi = _desc(smem + K_OFF + HALF_BYTES, 8 * 64 * 2);

            /* Pass 1: O_lo += P × V_lo (output DIM cols 0-63) */
            if (tid == 0) {
                _mi(mma_bar, 1);
                uint64_t vd0 = _desc(smem + V_OFF, 8 * 64 * 2);
                for (int kk = 0; kk < 8; kk++) {
                    uint64_t pd = (kk < 4) ? p_lo + (uint64_t)((kk * MMA_K * 2) >> 4)
                                           : p_hi + (uint64_t)(((kk-4) * MMA_K * 2) >> 4);
                    uint64_t vd = vd0 + (uint64_t)((kk * MMA_K * 64 * 2) >> 4);
                    _mma(O_lo, pd, vd, idesc_pv, true);  /* accumulate */
                }
                _commit(mma_bar);
            }
            _mw(mma_bar, 0);
            __syncthreads();

            /* Pass 2: O_hi += P × V_hi (output DIM cols 64-127) */
            if (tid == 0) {
                _mi(mma_bar, 1);
                uint64_t vd0 = _desc(smem + V_OFF + HALF_BYTES, 8 * 64 * 2);
                for (int kk = 0; kk < 8; kk++) {
                    uint64_t pd = (kk < 4) ? p_lo + (uint64_t)((kk * MMA_K * 2) >> 4)
                                           : p_hi + (uint64_t)(((kk-4) * MMA_K * 2) >> 4);
                    uint64_t vd = vd0 + (uint64_t)((kk * MMA_K * 64 * 2) >> 4);
                    _mma(O_hi, pd, vd, idesc_pv, true);  /* accumulate */
                }
                _commit(mma_bar);
            }
            _mw(mma_bar, 0);
            __syncthreads();
        }
    }

    /* ---- Finalize: O = O / rowsum ---- */
    _fb(); __syncthreads(); _fa();

    uint32_t o_lo_r[64], o_hi_r[64];
    _ld64(o_lo_r, O_lo);
    _wl();
    _ld64(o_hi_r, O_hi);
    _wl();

    float inv_sum = fast_rcp(rowsum);
    for (int i = 0; i < 64; i++) {
        o_lo_r[i] = __float_as_int(__int_as_float(o_lo_r[i]) * inv_sum);
        o_hi_r[i] = __float_as_int(__int_as_float(o_hi_r[i]) * inv_sum);
    }

    /* Write O to global */
    int seq_stride = H * DIM;
    for (int i = 0; i < 64; i++) {
        int row, col;
        tmem_coords_128(wid, lid, i, row, col);
        int gRow = q_block_id * BLOCK_Q + row;
        if (gRow < len_q && col < DIM) {
            int idx = (batch_id * S + gRow) * seq_stride + head_id * DIM + col;
            O_global[idx] = __float2bfloat16(__int_as_float(o_lo_r[i]));
        }
    }
    for (int i = 0; i < 64; i++) {
        int row, col;
        tmem_coords_128(wid, lid, 64 + i, row, col);
        int gRow = q_block_id * BLOCK_Q + row;
        if (gRow < len_q && col < DIM) {
            int idx = (batch_id * S + gRow) * seq_stride + head_id * DIM + col;
            O_global[idx] = __float2bfloat16(__int_as_float(o_hi_r[i]));
        }
    }

    /* Cleanup */
    __syncthreads();
    if (wid == 0) {
        _dealloc(S_tmem, 128);
        _dealloc(O_lo, 64);
        _dealloc(O_hi, 64);
    }
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

static int json_int(const char*j,const char*k,int fb){
    if(!j)return fb;const char*p=strstr(j,k);if(!p)return fb;
    p+=strlen(k);while(*p&&(*p=='"'||*p==':'||*p==' '))++p;
    int v=0;while(*p>='0'&&*p<='9'){v=v*10+(*p-'0');++p;}return v?v:fb;}

static int parse_int_env(const char*n,int fb){
    const char*v=getenv(n);return(v&&*v)?atoi(v):fb;}

extern "C" int kernel_run(
    __nv_bfloat16** inputs, int num_inputs,
    __nv_bfloat16** outputs, int num_outputs,
    int n, cudaStream_t stream)
{
    /* TMA uses cuTensorMapEncodeTiled directly (linked with -lcuda) */

    const char*cj=getenv("CUDA_EXEC_CONFIG_JSON");
    int B=parse_int_env("CUDA_EXEC_PARAM_BATCH_SIZE",0);
    int S=parse_int_env("CUDA_EXEC_PARAM_SEQ_LEN",0);
    int H=parse_int_env("CUDA_EXEC_PARAM_NUM_HEADS",0);
    int D=parse_int_env("CUDA_EXEC_PARAM_HEAD_DIM",0);
    if(!B)B=json_int(cj,"batch_size",0);
    if(!S)S=json_int(cj,"seq_len",0);
    if(!H)H=json_int(cj,"num_heads",0);
    if(!D)D=json_int(cj,"head_dim",0);
    if(!D)D=128;if(!H)H=16;
    if(!S&&!B&&n>0){B=1;S=n/(H*D);}
    if(!B||!S)return -1;

    /* Parse causal flag */
    bool causal = false;
    {
        const char*cv=getenv("CUDA_EXEC_PARAM_CAUSAL");
        if(cv && (strcmp(cv,"true")==0||strcmp(cv,"1")==0||strcmp(cv,"True")==0)) causal=true;
        if(!causal && cj) {
            const char*p=strstr(cj,"causal");
            if(p){p+=6;while(*p&&(*p=='"'||*p==':'||*p==' '))p++;
                if(strncmp(p,"true",4)==0) causal=true;}
        }
    }
    if(D!=128)return -2;

    __nv_bfloat16*Q=inputs[0],*K=inputs[1],*V=inputs[2];
    __nv_bfloat16*O=outputs[0];

    int seq_stride=H*D;
    cuuint64_t gd[2]={(cuuint64_t)seq_stride,(cuuint64_t)(B*S)};
    cuuint64_t gs[1]={(cuuint64_t)(seq_stride*2)};
    cuuint32_t bx[2]={64,(cuuint32_t)BLOCK_Q};
    cuuint32_t bxkv[2]={64,(cuuint32_t)BLOCK_KV};
    cuuint32_t el[2]={1,1};

    CUtensorMap tQ,tK,tV,tO;
    CUresult r;
    r=cuTensorMapEncodeTiled(&tQ,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)Q,gd,gs,bx,el,
       CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
       CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r)return -3;
    r=cuTensorMapEncodeTiled(&tK,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)K,gd,gs,bxkv,el,
       CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
       CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r)return -4;
    r=cuTensorMapEncodeTiled(&tV,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)V,gd,gs,bxkv,el,
       CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
       CU_TENSOR_MAP_L2_PROMOTION_L2_128B,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r)return -5;
    r=cuTensorMapEncodeTiled(&tO,CU_TENSOR_MAP_DATA_TYPE_BFLOAT16,2,(void*)O,gd,gs,bx,el,
       CU_TENSOR_MAP_INTERLEAVE_NONE,CU_TENSOR_MAP_SWIZZLE_128B,
       CU_TENSOR_MAP_L2_PROMOTION_NONE,CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);
    if(r)return -6;

    int num_blocks=B*H*cdiv(S,BLOCK_Q);
    cudaFuncSetAttribute(fa_tcgen05,cudaFuncAttributeMaxDynamicSharedMemorySize,SMEM_TOTAL);
    int max_kv = causal ? 0 : cdiv(S, BLOCK_KV);  /* 0 = per-block causal limit */
    fa_tcgen05<<<num_blocks,THREADS,SMEM_TOTAL,stream>>>(
        O,B,S,H,S,S,causal,tQ,tK,tV,tO);
    /* Ensure kernel completes before harness calls kernel_run again */
    cudaStreamSynchronize(stream);
    return 0;
}
