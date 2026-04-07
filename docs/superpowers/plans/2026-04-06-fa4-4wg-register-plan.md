# FA4 4-WG Register Allocation Plan

## 寄存器分配（按 AVO 方案）

| WG | 角色 | 线程数 | 寄存器/线程 | 总寄存器 |
|----|------|--------|------------|---------|
| WG0 | Producer (TMA load) | 128 | **48** | 6,144 |
| WG1 | Consumer A (QK+softmax+PV) | 128 | **192** | 24,576 |
| WG2 | Consumer B (QK+softmax+PV) | 128 | **192** | 24,576 |
| WG3 | Correction (O rescale + TMA S2G) | 128 | **80** | 10,240 |
| **合计** | | **512** | | **65,536** |

## setmaxnreg 指令

入口寄存器数 = 128（`__launch_bounds__(512, 1)` → 65536/512）

```
WG0: setmaxnreg.dec.sync.aligned.u32 48   → 释放 (128-48)×128  = 10,240
WG3: setmaxnreg.dec.sync.aligned.u32 80   → 释放 (128-80)×128  = 6,144
WG1: setmaxnreg.inc.sync.aligned.u32 192  → 获取 (192-128)×128 = 8,192
WG2: setmaxnreg.inc.sync.aligned.u32 192  → 获取 (192-128)×128 = 8,192

验证: 释放 = 10240 + 6144 = 16,384
      获取 = 8192 + 8192  = 16,384  ✓
```

**重要**：不是 512 个线程每个都用 128 个寄存器！
- Consumer（WG1/WG2）用 **192** 个寄存器
- Producer（WG0）用 **48** 个寄存器
- Correction（WG3）用 **80** 个寄存器

## Consumer 寄存器需求（192 budget）

```
O_acc[64]     = 64 regs   (PV 累加器，跨 KV 迭代持久)
S_acc[64]     = 64 regs   (QK 输出，每次迭代重用)
P_packed[32]  = 32 regs   (softmax 后 bf16 打包)
softmax scratch = ~20 regs (rv[], temps, rowmax, rowsumexp)
descriptors     = ~8 regs  (dq, dk, dv)
loop vars       = ~4 regs  (k_stage, v_stage, kv_id)
Total: ~192 regs → 完全放入 192 budget
```

## SM90 ptxas 的限制

ptxas 对整个 kernel 统一分配寄存器。它不理解 setmaxnreg 后不同 WG 有不同的寄存器数量。
- 入口 128 regs 必须能容纳所有代码路径的编译结果
- Consumer 需要 ~170 regs → 超过 128 → ptxas 要么 spill 要么拒绝编译
- setmaxnreg 被忽略（C7507 warning）

## 解决方案

### 方案 A：SMEM Staging（当前在实现）
- QK GEMM 前把 O_acc[64] 存到 SMEM scratch
- softmax+packP 后恢复 O_acc
- Peak live regs = max(S+scratch≈100, O+P≈96) < 128
- 代价：失去 QK/PV overlap，16 条 SMEM 指令/迭代

### 方案 B：Binary Patching（待尝试）
- 不用 __launch_bounds__，让 ptxas 分配 166 regs（0 spills）
- 手动 patch cubin 的 REGCOUNT 为 128
- 手动插入 USETMAXREG SASS 指令
- 风险：producer 代码可能使用 >48 的物理寄存器号

### 方案 C：分函数编译
- Producer/Correction 作为单独的 __noinline__ 函数（低寄存器）
- Consumer 作为主函数体（高寄存器）
- 挑战：wgmma pipeline crossing function boundary

## 当前状态
- 128 regs, 36 bytes spill（从 1444 bytes 大幅降低）
- 正确性：prologue 100% match 3-WG，mainloop 有 bug 待修
- 性能：153-166 TF（远低于 3-WG 548 TF）
- 根因：setmaxnreg 未生效 + 无 QK/PV overlap + SMEM staging 开销
