# FlashInfer SM80 (A100) MLA FP8 KV Cache 支持 — 4 周开发计划

> 目标：仿照 [PR #3694](https://github.com/flashinfer-ai/flashinfer/pull/3694)（SM90/Hopper MLA FP8 KV cache）的思路，
> 为 FA2 backend 的 MLA kernel（`include/flashinfer/attention/mla.cuh`）添加 SM80（A100）上的
> FP8 (e4m3) KV cache 支持。
>
> 计划总时长 4 周：**前 2 周为 CUDA / C++ / FlashInfer 学习**（面向零基础），**后 2 周为开发、测试与提交 PR**。

---

## 实施状态与核心结论（已完成，回填实测）

> 本文档最初是开发前的规划，下面这段是**开发完成后**根据实际实现与 A100 实测回填的结论摘要；
> 后续各阶段（Week 3/4）也已按实际发生的情况更新。

- **功能已实现并验证**：`mla.cuh` FA2 路径支持 FP8(e4m3) KV cache，A100(SM80) 上
  **155 个 FP8 精度用例全过、1350 个 BF16 回归用例零回归**。csrc/绑定层零改动（复用 PR #3694
  已铺好的 `ckv_scale`/`kpe_scale` 管道），实质改动集中在 `mla.cuh` + `_core.py` guard + 测试参数化。
- **一个与最初预期相反的关键发现**：**MLA decode 是 compute-bound，不是 memory-bound**，
  所以 FP8 KV **在 A100 上不会带来 decode 加速**（实测 0.86–0.96x，即回退 4–14%），
  即便做了"prefetch 与 QK/PV 重叠"的流水优化也只从 0.83–0.94x 抬到 0.86–0.96x；
  后续的 scale 折叠优化（Week 3 Day 9）再收窄至 **0.87–0.88x**，消融分析（§0.6.9）表明
  这已接近 staging 方案在 SM80 的结构性上限（寄存器路径 dequant 实验为负结果，已回退）。
  这**推翻了最初计划里"长序列 FP8 有正吞吐收益"的假设**（见 §0.6 的 roofline 分析与实测）。
- **FP8 对 MLA 的真实价值是"KV cache 显存精确减半"**（容量翻倍 → 更大 batch / 更长 context），
  即**可行性/容量**收益：让原本 OOM 的超长 context / 超大并发能跑、同预算装更长上下文、省卡降 TP。
  这与"给 A100 省显存"的原始诉求吻合。
  ⚠️ **但这不等于系统吞吐翻倍**：实测固定显存预算下 FP8-batch-2B vs BF16-batch-B 的吞吐只有
  **~0.85x（0.68–1.09x）**，因为 MLA decode 是 compute-bound——算力早已打满、没有空闲让多出的
  batch 去填，加 batch 只是线性堆延迟。"容量翻倍→吞吐翻倍"只在 memory-capacity-bound 且算力
  有空闲时成立（GQA），MLA 两个前提都不满足（见 §0.6.6）。
- **但有一个真实工况 FP8 能兑现吞吐**：**内存受压、BF16 KV 持续 retract（抢占后重算）** 时
  （如输入 1k/输出 32k、大 batch），FP8 减半 KV → 越过显存悬崖 → 避免重算浪费 → 有效吞吐上升；
  且**原生 fp8-kv 比"手动 python dequant 成 bf16 再喂 flashinfer"少一整趟 HBM 往返 + 临时显存**，
  收益更满（sglang 实测手动方案 +~5%；见 §0.6.7）。**本分支原生 fp8-kv 已在 4 机 DeepSeek-R1
  端到端跑通并坐实：retract 区（并发 256、1k-in/8k-out）输出吞吐 +29.9%、TTFT −89%、retract
  440→0；而显存富余稳态（并发 64）则 −4.1%，与 §0.6.6 预言一致——详见 §0.6.8**。
- **对照实证**：同机同法测 GQA decode，FP8 **加速 1.2–1.6x**（因为 GQA memory-bound）——
  正好反衬出 MLA 的 compute-bound 特性（§0.6）。

---

## 0. 背景与现状分析

### 0.1 GQA / MHA 在 SM80 上已经支持 FP8 KV cache

普通 GQA/MHA 的 FA2 backend（prefill + decode）在 SM80 上已有完整的 FP8 KV cache 支持：

- **Kernel 层**（`include/flashinfer/attention/prefill.cuh`）：
  - `KernelTraits::USE_KV_REPACK = (sizeof(DTypeKV) == 1) && ...`（prefill.cuh:212）；
  - KV 以 FP8 原样载入 shared memory，计算前由 `repack_fp8_tile_to_bf16()`（prefill.cuh:1016）
    **整 tile 反量化到 BF16 staging buffer**，之后走标准 16-bit `ldmatrix` + `mma.sync` 路径；
  - 原因：**SM80 没有 FP8 Tensor Core 指令**（FP8 MMA 从 SM89/SM90 才有），FP8 只能省显存和
    gmem 带宽，计算前必须升格为 16-bit。
- **Scale 处理**：GQA/MHA 用 per-tensor `k_scale`/`v_scale`，实现上直接把 `k_scale` 折叠进
  `sm_scale`（`flashinfer/decode.py:619`），`v_scale` 乘到输出上，kernel 内部反量化不带 scale。
- **测试**：`tests/attention/test_fp8_prefill.py`、`tests/attention/test_decode_fp8_calibration_scale.py`。

### 0.2 MLA 在 SM80 上不支持 FP8 KV cache

- FA2 的 MLA kernel（`include/flashinfer/attention/mla.cuh`，即 `BatchMLAPagedAttentionKernel`）
  **完全没有 FP8 路径**：swizzle stride、`load_kv` 的循环次数、P 矩阵（softmax 输出）的存储类型、
  PV MMA 全部按 16-bit DTypeKV 推导，喂 FP8 会静默算错；
- Python 侧 `flashinfer/mla/_core.py::plan()` 显式拦截：FP8 KV 仅允许 `fa3` backend + SM90
  （这正是 PR #3694 引入 SM90 支持时加的 guard）。

### 0.3 为什么 MLA 不能照抄 GQA 的 scale 折叠技巧

MLA 中 `ckv`（compressed KV，512 维）**既当 K(nope) 又当 V**，`kpe`（rope K，64 维）有独立 scale：

```
logits = q_nope · ckv_dequant + q_pe · kpe_dequant
       = ckv_scale * (q_nope · ckv_fp8) + kpe_scale * (q_pe · kpe_fp8)
```

两个不同的 scale 混在同一个 logits 里，**无法折叠进单一 `sm_scale`**。所以 PR #3694 的做法是把
`ckv_scale` / `kpe_scale` 作为显式参数传入 kernel，在 shared memory 反量化时逐 buffer 乘上。
SM80 也必须走这条路。

### 0.4 为什么 PR #3694 改了 12 个文件，而本次主要只改 `mla.cuh`

把 PR #3694 的 diff 按性质拆开：

| PR #3694 改动的文件 | 改动量 | 性质 |
|---|---|---|
| `mla_hopper.cuh` | +358 行 | **SM90 kernel 本体**——真正的 FP8 实现 |
| `mla_params.cuh` + `mla.cuh`（仅 variant 结构体） | +11 行 | 参数结构体新增 `ckv_scale`/`kpe_scale` 字段 |
| `batch_mla_sm90_run.cu` / `batch_mla_sm90_binding.cu` | ~8 行 | fa3 launcher 的 FFI 签名加 scale 参数 |
| `batch_mla_run.cu` / `batch_mla_binding.cu` | ~6 行 | **fa2（SM80）launcher 的签名也同步加了** |
| `flashinfer/mla/_core.py` | +119 行 | Python API 新增参数 + dtype 校验 + guard |
| `trace/templates/attention.py` + `tests/trace/*.json` | ~58 行 | API 签名变化 → trace 模板与快照重生成 |
| `tests/attention/test_deepseek_mla.py` | +684 行 | FP8 量化 helper + 正确性测试 |

两个关键点：

1. **PR #3694 是"第一次引入 FP8 MLA"，必须打通整条参数链路**：
   Python API → TVM-FFI binding → C++ launcher → `MLAParams` → kernel variant，每层都要加
   `ckv_scale`/`kpe_scale`，所以文件多——但 kernel 之外都是几行的"传参管道工程"。
2. **这条管道是 fa2/fa3 双 backend 共用的，且 PR 顺手把 fa2 侧也改完了**：
   Python wrapper 的 `run()` 对两个 backend 用同一套参数列表调 FFI，binding 签名必须一致，
   所以 `batch_mla_run.cu`（SM80 launcher）已经接收 scale 并塞进 `params`，`mla.cuh` 的
   `StandardAttention` variant 也已经在读这两个值——只是 SM80 kernel 的计算路径还没用它
   （默认 1.0，无害）。

本次开发是这条管道的**第二个消费者**：不改 API 签名 → csrc 四个文件、trace 模板/快照全部零改动；
不加新参数 → `_core.py` 只需放宽 guard 几行；测试框架现成 → 只需参数化 backend。
剩下的唯一实质工作就是在 `mla.cuh` 里实现 FP8 消费路径，对应 `mla_hopper.cuh` 那 +358 行的角色。

**结论：实际要改 `mla.cuh`（主要工作量）+ `_core.py`（几行 guard）+ 测试文件（参数化）。**

已就绪、可直接复用的部分汇总：

| 已就绪（无需改动） | 位置 |
|---|---|
| `ckv_scale` / `kpe_scale` 字段（默认 1.0） | `include/flashinfer/attention/mla_params.cuh` |
| SM80 launcher 已接收并设置 scale | `csrc/batch_mla_run.cu`、`csrc/batch_mla_binding.cu` |
| SM80 kernel 的 variant 已读取 scale（kernel 尚未使用） | `mla.cuh` 中 `StandardAttention::ckv_scale/kpe_scale` |
| Python `run()` 的 `ckv_scale`/`kpe_scale` 参数及合法性校验 | `flashinfer/mla/_core.py` |
| JIT 模块生成对 FP8 KV dtype 的渲染 | `flashinfer/jit/attention/modules.py`（`dtype_map_kv`） |
| FP8 参考测试（量化 helper + 误差对比逻辑） | `tests/attention/test_deepseek_mla.py` |

### 0.5 Shared memory 预算（决定支持范围的关键约束）

`mla.cuh` 的 `DISPATCH_SMEM_CONFIG` 按 smem 容量选 tile 配置，现有阈值是按 16-bit KV 精确算出的
结构体大小。FP8 路径新增 BF16 staging buffer 后：

| 配置档位 | BF16 大小 | FP8 大小（估算） | 结论 |
|---|---|---|---|
| 2-stage, CTA_TILE_KV=64（Hopper 227KB 档） | 221,696 B | ~229,888 B | H100 跑 fa2 也放得下 |
| 2-stage, CTA_TILE_KV=32（**A100 164KB 档**） | 147,968 B | **~152,064 B ✅** | **SM80 目标配置，放得下** |
| 1-stage, CTA_TILE_KV=16（100KB 档） | 92,672 B | ~102,912 B ❌ | **sm86/sm89 放不下，需明确报错** |

即：**FP8 MLA 的 FA2 路径支持 A100（SM80）和 Hopper，sm86/sm89 因 smem 不足不支持**，
Python 与 C++ 两层都要给出明确报错，不能静默算错。

### 0.6 FP8 KV 的性能特征：为什么 MLA 上 FP8 不加速（roofline 分析 + 实测）

这是本项目**最重要、也最反直觉的结论**，务必在动手前就理解清楚，否则会像最初计划那样错误地
预期"长序列 FP8 提速"。

#### 0.6.1 核心：GQA 是 memory-bound，MLA 是 compute-bound

FP8 KV 的加速逻辑只有一条：**当 kernel 卡在读 KV 的带宽上时**，把 KV 减半才能换来时间。
是否卡在带宽上，由 **算术强度（arithmetic intensity, AI = FLOP / 读取字节）** 相对 GPU 的
**ridge point** 决定：

- A100(SXM) ridge point ≈ FP16 tensor core 峰值 312 TFLOPS ÷ HBM 峰值 ~2.0 TB/s ≈ **156 FLOP/byte**；
- AI ≪ 156 → **memory-bound** → FP8 有效；AI ≫ 156 → **compute-bound** → FP8 无益。

decode 场景对每个 KV token 的 AI：

| 场景 | 每个 KV token 服务几个 query head | AI(BF16) | 相对 156 | 瓶颈 |
|---|---|---|---|---|
| **GQA/MHA decode** | 只 `group` 个（典型 4–8） | `group×2/2` ≈ **8** | ≪ 156 | **memory-bound** |
| **MLA decode（absorbed）** | **全部 128 个 head 共享一份 ckv latent** | ≈ **242** | ≫ 156 | **compute-bound** |

MLA 用 `≈ h × (576+512) × 2 / (576 × 2) ≈ h × 1.9`（`h` = head 数）估 AI。关键在于
**MLA 的 ckv(512)+kpe(64) 被所有 `h` 个 head 共享**——这是 MLA 的核心设计（MQA-like 存储、
MHA-like 表达力），它在**算法层面**就把 KV 带宽压力消掉了，代价是把 decode 变成了一个
高算术强度的 128-head GEMM。所以到 MLA 这里，FP8 已经**没有多余的带宽红利可吃**。

#### 0.6.2 各主流 MLA 模型的 head 数（决定 AI）

`h` 由模型决定，直接决定 FP8 是否可能有速度收益：

| 模型 | 注意力 | `num_attention_heads` | BF16 AI ≈ h×1.9 | vs 156 |
|---|---|---:|---:|---|
| DeepSeek-V2 / V3 / R1 | MLA | **128** | ~242 | compute-bound |
| GLM-5 / 5.1（`Glm5MoeDsa`） | MLA + DSA | ~64 | ~121 | 临界 |
| DeepSeek-V2-Lite | MLA | **16** | ~30 | 偏 memory-bound |
| GLM-4.5 / 4.6 | **GQA（非 MLA）** | 96q / 8kv | ~8 | memory-bound |

注意：**GLM-4.5/4.6 用的是 GQA 不是 MLA**，到 GLM-5 才用 MLA。主流 MLA 旗舰（V2/V3=128、
GLM-5.1≈64）的 head 数都足够高，落在 compute-bound 或临界区。本项目 benchmark 用的
`num_heads=128`/`16` 正是对应 V2/V3 满配与 V2-Lite。

#### 0.6.3 A100 实测（同机同法，`benchmarks/bench_deepseek_mla_fp8_kv.py`）

**MLA decode（fa2, page_size=64, ckv=512+kpe=64, q=bf16）**——优化后：

| batch | seq_len | heads | bf16 ms | fp8 ms | 加速比 | bf16 GB/s |
|--:|--:|--:|--:|--:|--:|--:|
| 64 | 4096 | 128 | 0.949 | 1.099 | **0.86x** | 337 |
| 64 | 16384 | 128 | 3.714 | 4.321 | **0.86x** | 330 |
| 16 | 16384 | 128 | 0.784 | 0.904 | **0.87x** | 391 |
| 16 | 32768 | 128 | 1.533 | 1.777 | **0.86x** | 397 |
| 16 | 65536 | 128 | 3.029 | 3.524 | **0.86x** | 400 |
| 16 | 1024 | 16 | 0.047 | 0.049 | **0.96x** | 412 |
| 64 | 16384 | 16 | 1.872 | 2.176 | **0.86x** | 646 |
| 16 | 32768 | 16 | 0.970 | 1.125 | **0.86x** | 623 |
| 16 | 65536 | 16 | 1.919 | 2.230 | **0.86x** | 630 |

**加速比对 seq_len 完全不敏感**：16384→32768→65536 全部钉在 **0.86x**（heads=128 与 16 皆然）。
这正是 roofline 的直接推论——MLA decode 的 AI 只由 head 数决定、**与 seq_len 无关**（FLOP 与
KV 字节随序列同步线性增长，比值不变），所以"加长序列 → 更 memory-bound → FP8 变有效"的直觉
**被实测证伪**。同时 BF16 带宽随 seq_len 变长是**平的**（不往峰值爬），说明卡在 compute 天花板、
不是带宽瓶颈。

**GQA decode 对照（group=8, head_dim=128, tensor-core decode）**：

| batch | seq_len | bf16 ms | fp8 ms | 加速比 | bf16 GB/s |
|--:|--:|--:|--:|--:|--:|
| 64 | 4096 | 0.621 | 0.400 | **1.55x** | 1728 |
| 64 | 16384 | 2.371 | 1.500 | **1.58x** | 1811 |

**铁证**：GQA BF16 带宽 ~1800 GB/s ≈ **89% HBM 峰值**（memory-bound）→ FP8 加速 1.5x；
MLA BF16 带宽只 ~340 GB/s ≈ **17% HBM 峰值**（compute-bound，SM 忙于算 GEMM）→ FP8 无益。
且 MLA 中 `num_heads` 越大带宽越低（128→330 vs 16→646），越 compute-bound、FP8 越吃亏，
完全符合 AI 随 head 数上升的分析。

> ⚠️ 一个易踩的坑：GQA FP8 若用**默认非 tensor-core decode**（`use_tensor_cores=False`）会走
> 标量 dequant 慢路径（实测 0.06x）。文章/上表的加速来自 `use_tensor_cores=True` + `k/v_scale`
> 的 tensor-core decode 路径。

#### 0.6.4 与 SM90（PR #3694）的关系澄清

- **SM90 的 MLA FP8 也不用 FP8 tensor core**：`mla_hopper.cuh` 注释明确写
  "WGMMA itself stays BF16xBF16 because Hopper does not have a mixed BF16xFP8 wgmma instruction"。
  因为是 bf16-Q × fp8-KV 的混合精度，硬件层面走不了 fp8 wgmma，只能反量化回 bf16——**和 SM80 做法本质相同**。
- 所以 **SM90 上 MLA FP8 decode 大概率也不加速**（同样 compute-bound）；PR #3694 本身只有
  正确性测试、无 benchmark，从未声称提速。SM90 相对 SM80 的唯一优势是 **warp specialization + TMA**
  让 repack 能和 WGMMA 真正流水重叠，因此**回退更小**，而非"因为有 fp8 tensor core 所以更快"。

#### 0.6.5 FP8 对 MLA 唯一能提速的窄窗口

FP8 只有在 **memory-bound 的 MLA 计算**里才提速，条件是 **un-absorbed（每 head 独立 K/V）
+ 短 q（低 AI）** 同时成立，例如：
- chunked prefill 的**小 chunk + 长 prefix**（读大量历史 latent、q chunk 短）；
- MTP / speculative decode 的 **verify**（q 是几个 draft token）。

sglang 的 `forward_mha.py`（un-absorbed，把 latent 解压成 per-head K/V 走标准 flash-attn）满足
"每 head 独立 KV"，但它用于 **prefill、q 通常很长**，默认仍是 compute-bound；只有落到上述短-q
子场景才 memory-bound、FP8 才有速度价值。而标准 decode 用 absorbed（compute-bound），
标准 prefill q 长（compute-bound），两头都不满足——所以这个窗口很窄，**FP8 对 MLA 的稳定价值
始终是省显存**。

#### 0.6.6 "省显存能否换来系统吞吐翻倍"——固定预算实测（否）

一个常见的乐观论断是："FP8 让 KV 减半 → 同显存塞 2x batch → 系统吞吐翻倍"。这只在
**memory-capacity-bound 且算力有空闲**时成立。为验证 MLA 是否满足，做固定 HBM 预算对照：
**BF16 batch B vs FP8 batch 2B**（两者 KV 占用相同），吞吐 = 每秒推进的请求-decode-step 数
（`benchmarks/bench_deepseek_mla_fp8_kv.py::fixed_budget_throughput`）：

| heads | seq_len | B→2B | bf16 kreq/s | fp8 kreq/s | 吞吐比 |
|--:|--:|--:|--:|--:|--:|
| 128 | 16384 | 8→16 | 18.5 | 17.7 | 0.96x |
| 128 | 16384 | 16→32 | 20.4 | 14.7 | 0.72x |
| 128 | 65536 | 8→16 | 4.15 | 4.54 | 1.09x |
| 128 | 65536 | 16→32 | 5.28 | 3.72 | 0.70x |
| 16 | 65536 | 8→16 | 9.65 | 7.17 | 0.74x |
| 16 | 65536 | 16→32 | 8.33 | 9.11 | 1.09x |

**结论：吞吐比在 0.68–1.09x 之间抖动、均值 ~0.85x，没有任何一点接近 2x。** 佐证：BF16 自身
batch 16→64（4x）时吞吐代理 `batch/latency` 不升反降（5.28→4.34 kreq/s），说明 batch=16 时
算力就已饱和。compute-bound 下延迟 ≈ k×(batch×seq×heads)，FP8 塞 2x batch → FLOP 翻倍 →
延迟翻倍 → 吞吐不变，再叠加 repack ~16% 开销故略亏。**因此"省显存"是可行性/容量收益（能跑、
能装更长、省卡），而非吞吐加速——后者在 SM80 MLA 上不成立。**

> 📌 **适用范围**：本节测的是"显存不吃紧的稳态"（BF16 也塞得下 batch）。当系统**内存受压、
> 会 retract/抢占**时,结论不同——见 §0.6.7,那里省显存能通过"避免重算浪费"兑现成真实吞吐。

#### 0.6.7 原生 fp8-kv vs 手动 python dequant：省一趟 HBM 往返 + retract 区才是兑现吞吐的场景

§0.6.6 的"无收益"是**显存充裕稳态**下的结论。真实 serving 常见的另一种工况——**大 batch + 长输出
（如输入 1k / 输出 32k）导致显存吃紧、BF16 KV 不断 retract（抢占后重算）**——FP8 反而有实打实的
吞吐收益。这里区分两条对比线：

**（1）为什么原生 fp8-kv 比"手动 python dequant"更快。**
老 flashinfer 的 MLA（SM80）只支持 qkv 全 bf16，用户若想省显存只能：把 KV 以 FP8 存 HBM →
计算前用 PyTorch **手动反量化成 BF16 写回 HBM** → 再喂给 flashinfer。每个 decode step 对被
attend 的 KV，其 HBM 流量为：

| 路径 | KV 的 HBM 流量 | 额外开销 |
|--|--|--|
| 手动 dequant（老） | ①读FP8 0.5x + ②写BF16 1.0x + ③读BF16 1.0x = **2.5x** | 独立 dequant kernel 启动 + 临时 BF16 buffer |
| **本分支原生**（bf16-q/fp8-kv） | 直接读 FP8 **0.5x**，在 smem 内 dequant → MMA | **无**写回、无重读、无临时 buffer、无额外 kernel |

原生路径省掉的正是 ②③ 那 **2.0x 的 KV HBM 流量 + 一次 kernel 启动 + 那块临时 BF16 显存**。

⚠️ **收益归因要准**：提升**不是** attention 计算变快了——原生 FP8 kernel 因 smem repack 反而比
纯 bf16 kernel 慢 ~16%（§0.6.3 的 0.86x）。收益全来自：**(a)** 消掉独立 dequant pass（它自身
读 0.5x + 写 1.0x = 1.5x KV 流量的 memory-bound 扫描）；**(b)** 消掉临时 BF16 buffer → 峰值显存
更低 → retract 区里再少几次抢占；**(c)** attention 内 KV 读取 1.0x→0.5x。粗算：老方案 ≈
`dequant_pass + bf16_attn`，原生 ≈ `1.16 × bf16_attn`；只要 `dequant_pass` 耗时 > 16% 的
`bf16_attn`（长序列下通常成立，因 dequant 是对全部历史 KV 的一趟扫描），原生即更快。

**（2）为什么这个场景 FP8 能兑现吞吐（而 §0.6.6 不能）。**
retract = 显存装不下 → 驱逐请求、之后**重算**，浪费的是算力。FP8 把 KV 减半 → 越过显存悬崖 →
**避免重算浪费** → 有效吞吐上去。这正是"可行性/容量收益"在**内存受压工况**下兑现成吞吐的形态：
**收益 = 帮你避开 retract 悬崖那一下**；悬崖以下（显存够）没用（§0.6.6），卡在悬崖上（此场景）
就是实打实的收益。原生实现峰值显存更低、又省 dequant 开销，能把这个收益吃得更满。

**实证参考**：用户在 sglang 上以"FP8 存储 + 手动 dequant"替换 BF16 存储（输入 1k/输出 32k、大
batch、原 BF16 持续 retract），**端到端吞吐 +~5%**。本分支的原生 fp8-kv 理论上应**≥ 该数值**
（多省一趟 HBM 往返 + 临时显存）。

⚠️ **边界**：幅度有界、依赖 workload 与 dequant 具体实现（整 cache / 分页 / 是否已融合）。坐实
数字需在 sglang 用本分支原生 fp8 MLA 对比手动方案、端到端跑 retract 区；单 kernel 微基准
（§0.6.3/0.6.6）复现不了 retract 动态。

#### 0.6.8 sglang 端到端实测（本分支原生 fp8 MLA，4 机 DeepSeek-R1）——坐实两条结论

§0.6.6/§0.6.7 的判据在单 kernel 微基准之外，已用本分支原生 fp8 MLA 在真实 serving 里跑通并量化。
两条结论——**"显存富余稳态无收益"**（§0.6.6）与 **"retract 区兑现吞吐"**（§0.6.7）——各设计
一组对照实验，同机同参同 seed、唯一变量是 `--kv-cache-dtype`（bf16 vs fp8_e4m3）。

**环境**：4 机 × 8×A100-80GB，DeepSeek-R1-0528-bf16，tp32 / dp4 / ep32 / dp-attention，
flashinfer 0.6.15（本分支，fa2 原生 fp8 MLA），NEXTN 投机解码（steps2 / topk1 / draft3），
`--max-running-requests 512`、`--mem-fraction-static 0.85`。fp8 使 KV pool 每 DP 组容量
**精确翻倍：287,336 → 574,672 token（同为 18.8 GB）**，直接印证"显存精确减半"。

**实验 A —— 显存富余稳态（并发 64，两臂均零 retract）**：负载 1k-in / 8k-out、128 条，
峰值 KV 需求 ~147K/组，只占 bf16 池 51%，两臂都不 retract → 隔离出纯 kernel 路径差异。

| 指标 | bf16 | fp8 e4m3 | fp8 变化 |
|---|---:|---:|---|
| Mean TPOT (ms) | 38.39 | 39.44 | **+2.7%（更慢）** |
| P99 TPOT (ms) | 43.74 | 45.53 | +4.1% |
| 输出吞吐 (tok/s) | 1550.3 | 1486.4 | −4.1% |
| retract 次数 | 0 | 0 | — |

结论：**与 §0.6.6 定量吻合**——显存不吃紧时 fp8 无收益、反而略亏。kernel 0.86x 的回退经
MoE 主导的整步稀释后（attention 仅占整步 ~17%），端到端只体现 ~2.7% TPOT 回退；余下约 1.4%
吞吐差来自 accept length 轻微下降（2.71→2.67）与采样噪声。

**实验 B —— retract 区（并发 256，bf16 越过显存悬崖、fp8 未越）**：负载 1k-in / 8k-out、
512 条，峰值 KV 需求 ~2.36M token，**落在 bf16 容量（1.15M）之上、fp8 容量（2.30M）之下**——
正是 §0.6.7 描述的悬崖工况。

| 指标 | bf16 | fp8 e4m3 | fp8 变化 |
|---|---:|---:|---|
| 输出吞吐 (tok/s) | 2611.8 | 3392.9 | **+29.9%** |
| 总时长 (s) | 1605.9 | 1236.2 | −23% |
| Mean TTFT (ms) | 63421 | 6673 | −89% |
| Mean TPOT (ms) | 74.27 | 68.03 | −8.4% |
| Max ITL (ms) | 265274 | 20222 | −92% |
| retract 次数 | 440 | 0 | — |

结论：**与 §0.6.7 定量吻合**——bf16 掉进 retract 悬崖（440 次抢占重算，Max ITL 高达 265 s 就是
重算卡顿），fp8 减半 KV 后恰好装下、零 retract → 输出吞吐 +29.9%、TTFT −89%。注意此处 fp8 的
TPOT 反而更低，是因为 bf16 的 retract 重算把 prefill 混进 decode 流抬高了尾延迟，这个收益远大于
fp8 kernel 本身 ~16% 的回退（后者只占整步 ~17% 的一小部分）。

**两实验合起来的完整叙事（关键）**：

| 工况 | KV 需求 vs 容量 | fp8 端到端吞吐 | 归因 |
|---|---|---:|---|
| 稳态（并发 64） | 两臂都富余 | **−4.1%** | 纯 kernel 回退（compute-bound） |
| retract 区（并发 256） | bf16 越悬崖、fp8 未越 | **+29.9%** | 消除 retract 重算浪费 |

即 **fp8 KV 是"显存悬崖开关"而非"decode 加速器"**：+29.9% 的全部来源是 retract 消除（§0.6.7），
一旦显存富余立刻回落到 §0.6.6 预言的 ~−3% 小幅回退。这与本文档最初被证伪的"长序列 FP8 提速"
假设形成闭环——收益永远是**容量**，只在容量恰好成为瓶颈（越悬崖那一下）时才转化为吞吐。

⚠️ **更极端 workload 的边界**：进一步把负载压到 1k-in / 32k-out / 并发 512（KV 需求 ~17M token，
**bf16 与 fp8 双双深度 oversubscribed**），预期 fp8 优势会**小于 +29.9%**——因为此时 fp8 也持续
retract，只是次数少于 bf16，收益从"零 retract vs 大量 retract"退化为"少量 retract vs 大量
retract"。该正式对比数据待补。

#### 0.6.9 kernel 内部耗时拆解与"寄存器路径 dequant"实验（负结果，已回退）

对已合入的 staging 实现（repack 到 BF16 staging，含 scale 折叠优化，0.87–0.88x）做消融拆解
（(64,16384,128) 配置，A100，改 kernel 得到时序真实/输出作废的变体）：

| 变体 | ms | 说明 |
|---|--:|---|
| bf16 基线 | 3.697 | |
| fp8 staging（现行实现） | 4.229 | 0.87x |
| − dequant ALU（repack 只搬运） | 3.925 | ALU 占 5.9% |
| − repack 全部（留 barrier） | 3.541 | smem 搬运占 10.4% |
| − barrier（地板） | **3.485** | **比 bf16 快 5.7%** |

地板快于 bf16 说明 KV 流量减半在 A100 上有真实的小幅 memory 收益，全部被 repack 的 ~17% 税吃掉。
据此实施了**寄存器路径 dequant 实验**（消灭 staging：QK 对 raw fp8 做 u16 视角 `ldmatrix` +
寄存器内 `fast_dequant`（Q 侧预先做 σ 维度置换，点积对收缩维置换不变）；PV 对 raw fp8 做
u16 `ldmatrix.trans` + PRMT 奇偶抽列 + 双 mma，epilogue 用 width-4 shuffle 恢复 o_frag 布局）。
**功能正确**（162 FP8 + 6126 BF16 全过），smem 降到 156160/115200B（A100 可开 CTA_TILE_KV=64
档），**但性能 0.73–0.78x，比 staging 的 0.87x 更差**，已回退（patch 存档：
`/tmp/fp8_register_dequant_experiment.patch`）。

**负结果的根因**（对后续尝试者的警示）：
1. **转换次数翻倍**：staging 的 repack 对每个 ckv 元素只转换一次、QK/PV 共享；寄存器路径 K、V
   各转一次（kpe 额外），总转换 ≈1.9x，还加了 PV 的 PRMT 抽列；
2. **依赖链插在 MMA 发射前**：dequant（PRMT→LOP3→HMUL2，~10 cycle 链）紧贴每条 HMMA 的
   B 操作数，而 kernel 寄存器已顶满 237–255（o_frag 128 个 f32 是大头），编译器没有余量做
   软件流水把链提前；
3. **A100 的 ALU:tensor 发射预算本来就紧**：每 k16 步 2 条 HMMA ≈16 cycle tensor 管线 vs
   ~18 cycle 的 ALU 流——内联 ALU 无法躲进 tensor 空隙，直接拉长关键路径。
   tier1/tier2 均测过（0.78x/0.74x），排除 tile 配置混杂。

**结论**：消融地板的 1.06x 只对"零转换"成立；SM80 没有 fp8→bf16 硬件指令，转换无处可逃，
staging 一次性转换 + 整 tile 向量化就是该架构下的合理局部最优。当前 0.87–0.88x 已接近结构性
上限；真正消掉这笔税需要 SM89+ 的 fp8 tensor core 或 Hopper 的异步流水（见 §0.6.4）。

**补充：为什么"staging 双缓冲 + repack/MMA 重叠"也不可行。** 一个自然的追问是：raw KV 已经
双缓冲（NUM_STAGES=2），能否给 BF16 staging 也加双缓冲，让 repack(T+1) 与 compute(T) 重叠？
答案是否定的，三层原因：

1. **容量**：tier2（CTA_TILE_KV=32）现占 152064B，第二份 staging（ckv 32KB + kpe 4KB）后
   188928B，超过 A100 每 block 上限 166912B。缩到 CTA_TILE_KV=16 能放下，但那一档
   `QK_SHARD=false`（两个 warpgroup 重复算 QK）、tile 减半使 barrier/循环开销翻倍，先亏一截。
2. **没有第二个执行流**：双缓冲只解决空间冲突；SM80 上 repack 没有独立执行单元，"重叠"只能
   是同一指令流内交错发射——这正是寄存器路径实验证伪的场景（ALU 塞进 MMA 流 → 0.87x 掉到
   0.74x），且交错 repack 比寄存器 dequant 管线压力**更大**：多出整趟 smem 搬运（LDS+2×STS，
   占 kernel 10.4%），抢的还是 MMA 阶段 `ldmatrix` 所在的 LSU 管线；寄存器已顶满 237–255，
   交错所需的 ~20 个临时寄存器必然 spill。瓶颈不是 buffer 不够，是没有闲置发射槽和寄存器去
   执行被重叠的工作。
3. **能造出第二执行流的方案都被结构堵死**：warp specialization（抽 warp 专职 repack）与
   CTA_TILE_Q=64 要求每 warpgroup 恰好 4 warp 的 MMA tiling 冲突；warpgroup ping-pong
   （wg0 算 tile T、wg1 repack T+1）要求每 warpgroup 独立持有全宽 512 维 o_frag = 256 个
   f32 寄存器（现靠 PV 维度切分才压到 128 个），寄存器文件放不下。这正是 §0.6.4 的反面：
   SM90 靠 TMA + producer/consumer warpgroup 的硬件异步机制才把 repack 藏进 WGMMA 背后，
   SM80 没有等价物。

**补充：Marlin 式 fast dequantization 在本场景的三层关系。** 有同学会问：Marlin 的
快速反量化技巧（`vec_dtypes.cuh::fast_dequant_f8f16x4`）在我们这里用不用得上？答案分三层：

1. **位操作转换本身——已经在用**。repack 的 `vec_cast<DTypeQ, DTypeKV>::cast<16>` 对
   `vec_size % 4 == 0` 直接分发到 `fast_dequant_f8f16x4`（每 4 个 fp8：1 个 `__byte_perm`
   重排 + 位与/移位提取符号尾数 + 2 个 `__hmul2` 乘指数 bias，~8 条指令出 4 个 bf16）。
   SM80 没有 fp8 的硬件 `cvt`（SM89+ 才有），这就是该架构最快的软件转换路径——消融测得
   dequant ALU 仅占 5.9%，正因为走的是它而非标量 `__nv_cvt_fp8_to_halfraw`。
2. **"融合进 GEMM mainloop"——试过了，就是上面的寄存器路径实验（负结果）**。Marlin 的
   杀手锏是把 dequant 内联进主循环、藏进 tensor core 的发射空隙，但其成立前提与本 kernel
   相反：Marlin 是 memory-bound 的 W4/W8 GEMM（tensor 管线大量空闲、权重 fragment 小），
   MLA decode 是 compute-bound（tensor 打满、o_frag 占 128 个 f32、寄存器顶格）——
   前提不成立，融合即负收益。
3. **"scale 融进 bias 乘法"——做了更彻底的版本**。Marlin 把 per-group scale 融进
   fast_dequant 本来就要做的 bias `__hmul2`；本实现（scale 折叠，见 Week 3 Day 9）把
   scale 整个挪出 repack（折进 Q + fp32 epilogue），repack 已是纯转换形态。
   剩下唯一未榨的尾部优化：把 bias 乘法（e4m3→bf16 固定的 2^120 指数修正）也照样挪出去
   （QK 侧拆 2^60 进 Q、2^60 进 sm_scale；PV 侧进 epilogue），预计 ~1–2%；但有数值边界
   （Q×2^60 极端值可能溢出 bf16 上界、PV 侧 2^-120 域小值可能被 FTZ 冲掉），属于
   "收益小、风险需论证"的 future work，暂不做。

#### 0.6.10 融合 fp8-kv kernel vs "fp8 存储 + 离线 dequant + bf16 kernel"（两段式）实测

一个自然的替代方案：KV cache 仍存 fp8（省显存），但不改 kernel——每个 decode step
先把整个 cache dequant 成 bf16，再走原生 bf16 MLA kernel。实测证明该方案全面劣于
本分支的融合 kernel。

**实验设置**（A100, fa2, page=64, q=bf16；脚本 `/tmp/bench_fp8_fused_vs_offline_dequant.py`）：
两条路径 KV 都以 fp8 驻留 HBM。两段式路径 dequant 到预分配 bf16 buffer
（`copy_` 转型 + `mul_` scale，timed loop 内零分配），`two-step` 列为 dequant+kernel
在同一 lambda 内端到端计时；两路输出 cos ≥ 0.99999、max diff ≤ 1.2e-4（数值等价）。

```
batch seq_len heads  kv GB | fused ms |  dequant bf16krnl two-step | speedup
   16    1024    16   0.01 |   0.0500 |   0.0775   0.0468   0.1174 |   2.35x
   64    1024    16   0.04 |   0.1553 |   0.2936   0.1423   0.4272 |   2.75x
   16    4096    16   0.04 |   0.1168 |   0.2935   0.1069   0.3914 |   3.35x
   64    4096    16   0.15 |   0.5502 |   1.1413   0.4863   1.6243 |   2.95x
   16   16384    16   0.15 |   0.4821 |   1.1417   0.4262   1.5641 |   3.24x
   64   16384    16   0.60 |   2.1279 |   4.5287   1.8690   6.3931 |   3.00x
   16   65536    16   0.60 |   2.1895 |   4.5282   1.9192   6.4446 |   2.94x
   64   65536    16   2.42 |   8.4371 |  18.1484   7.4068  25.5584 |   3.03x
   16    1024   128   0.01 |   0.0904 |   0.0775   0.0821   0.1543 |   1.71x
   64    1024   128   0.04 |   0.2905 |   0.2936   0.2568   0.5434 |   1.87x
   16    4096   128   0.04 |   0.2319 |   0.2936   0.2075   0.4937 |   2.13x
   64    4096   128   0.15 |   1.0788 |   1.1414   0.9451   2.0844 |   1.93x
   16   16384   128   0.15 |   0.8916 |   1.1418   0.7831   1.9226 |   2.16x
   64   16384   128   0.60 |   4.2276 |   4.5269   3.6986   8.2289 |   1.95x
   16   65536   128   0.60 |   3.4516 |   4.5269   3.0223   7.5534 |   2.19x
   64   65536   128   2.42 |  16.8305 |  18.1470  14.7122  32.8850 |   1.95x
```

**结论：融合 kernel 快 1.7–3.4x**（heads=128 约 2x，heads=16 约 3x），且省显存 3 倍。

1. **dequant 本身就 ≥ 融合 kernel 的全部耗时**。64×64K：dequant 18.1ms vs 融合
   kernel 16.8ms。离线 dequant 对整个 cache 一读一写再一读一写（copy 3B/elem +
   mul 4B/elem ≈ 7B/elem，实测 ~940 GB/s）；融合 kernel 只读 1B/elem，dequant
   藏在本来就 compute-bound 的流水线里几乎免费（§0.6.9 消融：ALU 仅 5.9%）。
2. **把离线 dequant 优化到理论极限也翻不了盘**：理想单 pass cast+scale kernel 为
   3B/elem，64×64K 约 5.4ms @1.4TB/s，两段式下限 14.7+5.4≈20.1ms，仍比融合慢 ~1.2x。
3. **heads=16 加速比更高（~3x）**：bf16 kernel 随 heads 变便宜，dequant 开销不变——
   KV 流量相对 attention 计算越"贵"，两段式越亏。
4. **两段式的显存代价是反向的**：fp8 原件 + bf16 副本需同时驻留（3B/elem），比纯
   bf16（2B/elem）还多，完全抵消 fp8 存储的初衷；融合路径只需 1B/elem。这也再次
   印证 §0.6.7 的结论——省一趟 HBM 往返 + 不留 bf16 副本，正是原生 fp8-kv 的价值。

---

## 1. Week 1 — C++ / CUDA 基础（学习周 1）

目标：掌握读懂 `mla.cuh` 所需的最小知识集。**不求全面，只学这个 kernel 用到的东西。**

### Day 1–2：C++ 模板与元编程（kernel traits 的语言基础）

学习内容：
- 类模板 / 函数模板、模板特化、非类型模板参数（`template <uint32_t N>`）；
- `constexpr` / `static constexpr`、`if constexpr`（kernel 里编译期分支的核心手段）；
- `std::is_same_v`、`std::conditional_t`（FP8/BF16 双路径共用一份代码的关键）；
- `union`、`alignas`（shared memory 复用与对齐）；
- 练习：读懂 `mla.cuh` 的 `KernelTraits` 和 `SharedStorageQKVO`（先不求懂 GPU 语义，只求懂模板语法）。

参考资料：
- [learncpp.com](https://www.learncpp.com/) 模板相关章节（Function templates / Class templates，入门最友好）；
- [cppreference.com](https://en.cppreference.com/)：`constexpr`、`if constexpr`、
  [`<type_traits>`](https://en.cppreference.com/w/cpp/header/type_traits)（重点 `std::is_same` / `std::conditional`）、
  [`union`](https://en.cppreference.com/w/cpp/language/union)、[`alignas`](https://en.cppreference.com/w/cpp/language/alignas)；
- 《C++ Templates: The Complete Guide, 2nd Edition》第 1–3 章（选读，遇到读不懂的模板语法时当工具书查）；
- 对照练习材料：本仓库 `include/flashinfer/attention/mla.cuh` 第 51–116 行
  （`SharedStorageQKVO` + `KernelTraits`）。

### Day 3–4：CUDA 编程模型与内存层级

学习内容：
- grid / block / warp / lane，SIMT 执行模型，`__syncthreads()` 语义；
- global memory / shared memory / register 三级层次，coalesced access，bank conflict；
- 异步拷贝 `cp.async`（对应本仓库 `cp_async::load_128b_async / commit_group / wait_group`）与
  多 stage 流水线（double buffering）——`mla.cuh` 主循环就是 2-stage 软件流水线；
- 练习：写一个 naive softmax kernel + 一个用 shared memory tile 的矩阵乘。

参考资料：
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)：
  第 1–5 章（Programming Model / Hardware Implementation / Performance Guidelines）
  及 "Asynchronous Data Copies" 一节（`cp.async` / `memcpy_async`）；
- 《Programming Massively Parallel Processors》(PMPP) 4th ed. 第 1–6 章（编程模型、内存层次、tiling）；
- NVIDIA 技术博客三篇经典入门：
  [How to Access Global Memory Efficiently in CUDA C/C++ Kernels](https://developer.nvidia.com/blog/how-access-global-memory-efficiently-cuda-c-kernels/)、
  [Using Shared Memory in CUDA C/C++](https://developer.nvidia.com/blog/using-shared-memory-cuda-cc/)、
  [CUDA Pro Tip: Increase Performance with Vectorized Memory Access](https://developer.nvidia.com/blog/cuda-pro-tip-increase-performance-with-vectorized-memory-access/)；
- [GPU-MODE 系列讲座](https://github.com/gpu-mode/lectures)（配 YouTube 录像，前 8 讲覆盖本日内容）；
- [NVIDIA Ampere GPU Architecture Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
  （A100 的 smem 容量、cp.async 硬件特性）；
- 对照练习材料：`include/flashinfer/cp_async.cuh` + `mla.cuh` 主循环（第 906–1070 行）。

### Day 5–6：Tensor Core、swizzle 与 FP8

学习内容：
- `mma.sync.m16n8k16` / `ldmatrix.m8n8x4` / `stmatrix` 指令语义，fragment 在 32 个 lane
  上的分布布局；
- shared memory swizzle（k128B / k64B permuted layout）：为什么 `ldmatrix` 需要 permute
  来避免 bank conflict；
- FP8 格式：e4m3 / e5m2 的数值范围，per-tensor 对称量化（`scale = amax / 448`），
  `__nv_fp8_e4m3`、`vec_cast` 批量类型转换；**Marlin 式 fast dequantization**
  （`vec_dtypes.cuh::fast_dequant_f8f16x4`：`__byte_perm` 重排 + 位操作提取 + bias
  `__hmul2`，SM80 无硬件 fp8 `cvt` 时的最快软件路径——repack 的 `vec_cast::cast<16>`
  底下就是它，见 §0.6.9 的三层关系分析）；
- **mma fragment 的 lane 级布局**（m16n8k16 的 A/B/C 各寄存器 ↔ (row, col) 映射）：
  不只是读懂 `compute_qk_` 需要，也是评估"寄存器路径 dequant"这类改造（§0.6.9）的基础；
  顺带理解**点积对收缩维置换不变**这一变换自由度（σ 置换技巧的数学依据）；
- **寄存器预算意识**：`ptxas -v` 看 kernel 的 registers/spill——本 kernel o_frag 占
  128 个 f32、总量 237–255 顶格，是否还有调度余量直接决定"能不能往 MMA 流里塞 ALU"（§0.6.9）；
- **roofline / 算术强度（arithmetic intensity）模型**：memory-bound vs compute-bound、
  ridge point 的概念——这是判断"FP8 减半 KV 带宽到底有没有速度收益"的唯一正确工具，
  务必配合 §0.6 一起理解；
- 关键认知（**已按实测更正**）：**SM80 无 FP8 MMA，FP8 KV 必须计算前反量化成 BF16**。
  FP8 省显存是**无条件**的（KV 存储精确减半）；但 FP8 是否**提速**取决于 kernel 是不是
  memory-bound——GQA/MHA decode 是（FP8 加速），**MLA decode 不是**（compute-bound，
  FP8 反而因多出 dequant 而小幅回退，见 §0.6）。**不要想当然地认为"带宽减半 = 加速"。**

参考资料：
- [PTX ISA 文档](https://docs.nvidia.com/cuda/parallel-thread-execution/) "Warp Level Matrix
  Multiply-Accumulate Instructions" 章节：`mma`（重点 m16n8k16 的 fragment 布局图）、
  `ldmatrix`、`stmatrix`（提示：URL 后加 `.md` 可得 Markdown 版）；
- NVIDIA 技术博客
  [Programming Tensor Cores in CUDA 9](https://developer.nvidia.com/blog/programming-tensor-cores-cuda-9/)
  （tensor core 入门概念）；
- CUTLASS/CuTe 文档（本仓库 `3rdparty/cutlass/media/docs/`，建议直接读源码树里的版本）：
  CuTe layout 系列与 shared memory swizzle / bank conflict 相关文档；
- 本仓库实现（swizzle 的最直接教材）：`include/flashinfer/permuted_smem.cuh`
  （`get_permuted_offset` / `SwizzleMode::k128B/k64B`）、`include/flashinfer/mma.cuh`
  （`mma_sync_m16n16k16_row_col_f16f16f32` 等 warp 级封装）；
- FP8 论文：[FP8 Formats for Deep Learning (arXiv:2209.05433)](https://arxiv.org/abs/2209.05433)
  （e4m3/e5m2 的定义与动机）；
- [CUDA Math API 文档](https://docs.nvidia.com/cuda/cuda-math-api/) 的 FP8 部分
  （`cuda_fp8.h`：`__nv_fp8_e4m3` 类型与转换 intrinsics）；
- 本仓库 FP8 转换实现：`include/flashinfer/vec_dtypes.cuh` 的 `vec_cast<bf16, fp8_e4m3>`
  与 `fast_dequant_f8f16x4`（Marlin 式位操作反量化，逐行读一遍，配合
  [MARLIN 论文 (arXiv:2408.11743)](https://arxiv.org/abs/2408.11743) 理解其原始使用场景
  ——memory-bound 权重反量化——再对照 §0.6.9 理解为什么同一技巧在 compute-bound 的
  MLA decode 里"转换函数用得上、mainloop 融合用不上"）；
- **roofline 模型**：[Roofline: An Insightful Visual Performance Model (Williams et al., CACM 2009)](https://dl.acm.org/doi/10.1145/1498765.1498785)
  或任意"roofline / arithmetic intensity"入门；配合本文档 §0.6 把 GQA/MLA 各自的 AI 算一遍，
  理解"为什么同样是 FP8 KV，GQA decode 加速而 MLA decode 不加速"。

### Day 7：FlashAttention 算法

学习内容：
- FlashAttention 的 tiling + online softmax：拿纸笔推一遍 m（running max）/ d（running sum）/
  o（running output）的增量更新公式；
- 对照 `mla.cuh` 的 `update_mdo_states_` / `normalize_d_` 把公式和代码对上。

参考资料：
- [Online normalizer calculation for softmax (arXiv:1805.02867)](https://arxiv.org/abs/1805.02867)
  （online softmax 原始论文，最短最清楚）；
- [FlashAttention (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135) 第 3 节算法部分；
- [FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691)
  （FA2 的工作划分方式，正是 fa2 backend 名字的由来）；
- 对照代码：`mla.cuh` 的 `update_mdo_states_`（第 383 行起）、`normalize_d_`（第 592 行起）。

**Week 1 验收**：能逐行解释 `mla.cuh` 的 `compute_qk_()` 里每条 `ldmatrix` / `mma_sync` 在做什么；
能解释 2-stage 流水线中 `wait_group<NUM_STAGES-1>` 为什么是 `<1>`。

---

## 2. Week 2 — FlashInfer 代码库与 MLA（学习周 2）

### Day 1：环境搭建与跑通

学习内容：

```bash
git clone https://github.com/ehuaa/flashinfer.git --recursive && cd flashinfer
pip install --no-build-isolation -e . -v
# 跑通 BF16 MLA 测试（JIT 首次编译较慢）
pytest tests/attention/test_deepseek_mla.py -k "not fp8" -x -q
```

- 理解 JIT 工作流：改 `include/**/*.cuh` → 重跑测试自动重编译，无需重装；
- 看一眼 `~/.cache/flashinfer/` 下生成的代码和 build.ninja。

参考资料：
- 本仓库 `CLAUDE.md`（Quick Start / How JIT Compilation Works / 环境变量参考，开发期最常用的一份文档）；
- 本仓库 `README.md` + [FlashInfer 官方文档](https://docs.flashinfer.ai/)（安装与 API 总览）；
- `.claude/skills/` 目录下的教程（add-cuda-kernel / benchmark-kernel / debug-cuda-crash）。

### Day 2：FlashInfer 分层架构

学习内容：
- 三层结构：`include/`（框架无关 kernel 模板）→ `csrc/`（TVM-FFI launcher）→
  `flashinfer/`（Python API + JIT codegen）；
- plan/run 两段式 API：`BatchMLAPagedAttentionWrapper.plan()`（JIT 选模块 + 调度）与 `run()`；
- 跟踪一次完整调用链：`_core.py::run()` → `batch_mla_run.cu::BatchMLAPagedAttentionRun`
  → `mla.cuh::BatchMLAPagedAttention`（launcher）→ `BatchMLAPagedAttentionKernel`。

参考资料：
- FlashInfer 系统论文：[FlashInfer: Efficient and Customizable Attention Engine for LLM
  Inference Serving (arXiv:2501.01005)](https://arxiv.org/abs/2501.01005)（MLSys 2025，
  讲清楚 plan/run、JIT、统一抽象的设计动机）；
- 本仓库 `CLAUDE.md` "Architecture: JIT Compilation System" 一节（JitSpec 三层结构）；
- [TVM-FFI 文档](https://tvm.apache.org/ffi/)（`TVM_FFI_DLL_EXPORT_TYPED_FUNC` 绑定机制，浏览即可）；
- 对照代码：`flashinfer/mla/_core.py`、`csrc/batch_mla_run.cu`、`csrc/batch_mla_binding.cu`、
  `flashinfer/jit/attention/modules.py::gen_batch_mla_module`。

### Day 3：MLA 算法（含两种计算方式）与 paged KV cache

学习内容：
- MLA（Multi-head Latent Attention）：ckv（512 维 compressed KV）+ kpe（64 维 rope K）、
  q_nope / q_pe 的双段 QK、为什么 ckv 既是 K 又是 V；
- **MLA 的两种数学等价、性能特征却相反的计算方式**（理解 §0.6 的前提，务必搞懂）：
  - **absorbed（矩阵吸收 / MQA-like）**：把 `W_uk` 吸收进 `W_q`、`W_uv` 吸收进 `W_o`，
    attention 直接在压缩的 ckv latent 上算，**128 个 head 共享一份 ckv**。用于 **decode**
    （q=1，省显存省带宽），**算术强度高 → compute-bound**。这就是 `mla.cuh` 这个 kernel 走的路径；
  - **un-absorbed（MHA 方式）**：用 `kv_b_proj` 把 latent 解压回每个 head 的完整 K/V
    （head_dim 192/128），走标准 flash-attention。用于 **prefill/extend**（q 长，计算量比
    absorbed 小），**长 q 时 compute-bound；短 q 时才 memory-bound**（见 §0.6.5）。
    sglang `forward_mha.py` 就是这条路径；
- paged KV cache：page_table / kv_indices / kv_indptr 的寻址方式；
- 对照 `load_kv()` 里 `block_size.divmod(packed_block_iter, q, r)` 理解 page 寻址。

参考资料：
- [DeepSeek-V2 论文 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) 第 2.1 节
  （MLA 定义，重点看矩阵吸收后 attention 只作用于 c^KV 和 k^R 的形式——这正是
  kernel 里 ckv/kpe 两个 cache 的来源、以及 absorbed 方式的数学基础）；
- absorbed vs un-absorbed 的工程解释：sglang / vLLM 的 MLA 实现博客与源码
  （sglang `python/sglang/srt/models/deepseek_common/attention_forward_methods/` 目录下
  `forward_mha.py`(un-absorbed prefill) 与 absorbed decode 路径对照）；
- [vLLM / PagedAttention 论文 (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
  （paged KV cache 的动机与结构）；
- [FlashInfer 官方文档](https://docs.flashinfer.ai/) 的 MLA API 页
  （`BatchMLAPagedAttentionWrapper` 的参数语义：qo_indptr / kv_indptr / kv_indices / kv_len_arr）；
- 各模型 head 数（决定算术强度）：DeepSeek-V2/V3 config（128 heads）、
  [DeepSeek-V2-Lite](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)（16 heads）、
  GLM-5/5.1 `Glm5MoeDsa` config（~64，MLA）、GLM-4.5/4.6（96q/8kv，**GQA 非 MLA**）；
- 对照代码：`tests/attention/test_deepseek_mla.py` 中任一 BF16 测试的数据构造部分
  （最直观的"MLA 输入长什么样"）。

### Day 4–5：精读 `mla.cuh`（本次要改的文件）

学习内容——逐函数读，画出两张图（这两张图是 Week 3 开发的施工图）：

1. **Shared memory 布局图**：`q_smem_nope / q_smem_pe / ckv_smem[2] / kpe_p_smem[2] /
   m_wg / d_wg` 与 `o_smem` 的 union 复用关系，各配置档位的字节数；
2. **主循环时序图**：mask 循环 / no-mask 循环 / last-tiles 循环三段中，
   `load_kv`(cp.async) / `wait_group` / `__syncthreads` / QK / softmax / PV 的先后关系，
   哪个 stage buffer 在何时被谁读写。

注意点：
- `QK_SHARD`（两个 warpgroup 分摊 KV 维度）下 P 矩阵要经 `kpe_p_smem` 中转 + allgather；
- `kpe_p_smem` 一物两用：QK 阶段存 kpe，PV 阶段复用为 P —— **这是 FP8 改造的坑点之一**。

参考资料：
- 主教材就是源码本身：`include/flashinfer/attention/mla.cuh`（约 1100 行，推荐阅读顺序：
  `KernelTraits` → `load_q`/`load_kv` → `compute_qk_` → `update_mdo_states_`
  → `compute_mla_pv` → `write_o` → 主循环 `BatchMLAPagedAttentionKernel`）；
- 辅助工具：`FLASHINFER_JIT_DEBUG=1` + `cuda-gdb`，或在 kernel 里临时 `printf`
  （用法见 `.claude/skills/debug-cuda-crash/skill.md`）；
- 卡在 fragment 布局时回查 Week 1 Day 5–6 的 PTX ISA `mma`/`ldmatrix` 布局图；
- 卡在 swizzle 偏移时回查 `include/flashinfer/permuted_smem.cuh`，手算一两个
  `get_permuted_offset` 的例子。

### Day 6：精读 FP8 参考实现（本次开发的两份"抄作业"对象）

学习内容：
- `prefill.cuh` 的 FP8 路径：`USE_KV_REPACK` traits、`repack_fp8_tile_to_bf16()`
  （每线程读 16B=16 个 fp8，`vec_cast` 转 16 个 bf16，写两个 16B）；
- **PR #3694 的完整 diff**，重点：
  - `mla_hopper.cuh` 的 `USE_KV_REPACK` / BF16 staging buffer / `repack_fp8_kv_to_bf16()`；
  - **KPE raw FP8 buffer 为什么必须换 k64B swizzle**（FP8 下一行只有 64B，k128B swizzle
    会产生行间地址冲突 —— Hopper PR 踩过的坑，SM80 会原样遇到）；
  - dtype-aware 的 load 循环边界（FP8 一个 b128 装 16 个元素，16-bit 假设的循环会超界 2 倍）；
  - "P is always DTypeQ-typed" 的处理。

参考资料：
- [PR #3694](https://github.com/flashinfer-ai/flashinfer/pull/3694) 页面（描述 + review 讨论）；
  本地看 diff：
  ```bash
  git fetch origin pull/3694/head:pr3694
  git log pr3694 --oneline -3          # 首个 commit 为 2fa5099
  git diff 2fa5099^..pr3694 -- include/flashinfer/attention/mla_hopper.cuh
  ```
- `include/flashinfer/attention/mla_hopper.cuh`（合入后的 SM90 实现，注释写明了
  swizzle 冲突、smem 预算、barrier 配对等设计理由，**最重要的一份参考**）；
- `include/flashinfer/attention/prefill.cuh` 第 1009–1042 行（`repack_fp8_tile_to_bf16`，
  SM80 上 GQA/MHA 的 FP8 dequant idiom）；
- `tests/attention/test_deepseek_mla.py` 的 FP8 测试段（量化 helper、误差阈值的设定依据）；
- **本分支后续演进（合入后再读）**：commit 950c904d（scale 折叠：为什么 scale 能整体挪出
  repack、Q 单独 commit group 的精确等待技巧）与 §0.6.9（消融拆解方法论 + 寄存器路径
  dequant 负结果 + 双缓冲/Marlin 分析——一份"哪些优化方向已被定量排除"的地图）。

### Day 7：缓冲与自查

学习内容：
- 用 `FLASHINFER_LOGLEVEL=3` 跑一遍 MLA 测试观察 API 输入输出；
- 自查清单（答不上来就回头补对应内容）：
  1. FP8 下 `UPCAST_STRIDE_CKV` 变成多少？为什么？（512/16=32）
  2. `kpe_p_smem` 若保持 FP8 元素类型，P 放不放得下？（放不下，P 必须 16-bit）
  3. last-tiles 循环和前两个循环在同步上有什么区别？（没有迭代间 barrier）

参考资料：
- 本仓库 `CLAUDE.md` "API Logging with @flashinfer_api" 与 "Debugging" 两节；
- 前 6 天所有材料（按自查结果回补）。

**Week 2 验收**：能不看代码画出上面两张图；能口述 PR #3694 在 Hopper 上做了哪 5 类改动。

---

## 3. Week 3 — 开发（`mla.cuh` kernel 改造 + Python guard）

总体思路与 PR #3694 同构：**FP8 存 smem → 计算前整 tile 反量化（乘 scale）到 BF16 staging →
QK/PV MMA 照常走 BF16**。SM80 没有 producer/consumer warpgroup 分工，256 线程全员参与
repack，比 Hopper 版简单。

> **实际实现对照**：Day 1–5 的 kernel 改造与计划基本一致，均已落地并通过编译/正确性验证。
> 几处值得记录的实际细节：
> - **A100 实测走 tier2**（`NUM_STAGES=2, CTA_TILE_KV=32, QK_SHARD=true`）——不是 tier1(CTA_TILE_KV=64)，
>   因为 A100 的 164KB smem 装不下 tier1；
> - **smem 精确尺寸已用 nvcc `static_assert` 核验**：FP8 `sizeof(SharedStorage)` 恰为
>   tier2=152064B、tier1=229888B，与 `DISPATCH_SMEM_CONFIG` 阈值精确吻合（不是估算）；
> - 提交历史见分支 `dev-plan-sm80-mla-fp8` 的 commit `feat: FP8 KV cache support for SM80 (A100) FA2 MLA`。

### Day 1：KernelTraits + SharedStorage（改 `mla.cuh`）

- `KernelTraits` 新增：
  ```cpp
  static constexpr bool USE_KV_REPACK = std::is_same_v<DTypeKV_, __nv_fp8_e4m3>;  // 精确匹配，不用 sizeof==1
  static_assert(!USE_KV_REPACK || (HEAD_DIM_CKV_ == 512 && HEAD_DIM_KPE_ == 64), "...");
  // BF16 staging 的 stride 一律按 DTypeQ 推导
  static constexpr uint32_t UPCAST_STRIDE_CKV_BF16 = HEAD_DIM_CKV / upcast_size<DTypeQ_>();
  static constexpr uint32_t UPCAST_STRIDE_KPE_BF16 = HEAD_DIM_KPE / upcast_size<DTypeQ_>();
  // UPCAST_STRIDE_P 从按 DTypeKV 改为按 DTypeQ 推导（BF16 路径下数值不变）
  // dtype-aware 的 load 边界（照搬 Hopper PR）
  static constexpr uint32_t CKV_B128_PER_ROW = HEAD_DIM_CKV * sizeof(DTypeKV_) / 16;
  static constexpr uint32_t KPE_B128_PER_ROW = HEAD_DIM_KPE * sizeof(DTypeKV_) / 16;
  // → INNER_LOADS_CKV / INNER_LOADS_KPE / LANES_PER_ROW_KPE
  // KPE raw FP8 换 k64B swizzle（k128B 在 <8 个 b128/行时产生地址冲突）
  static constexpr SwizzleMode SWIZZLE_MODE_KPE_RAW =
      (USE_KV_REPACK && KPE_B128_PER_ROW < 8) ? SwizzleMode::k64B : SwizzleMode::k128B;
  ```
- `SharedStorageQKVO`：
  - `kpe_p_smem` 改为 union：`{ DTypeKV kpe[CTA_TILE_KV*HEAD_DIM_KPE]; DTypeQ p[CTA_TILE_KV*CTA_TILE_Q]; }`
    （P 是 softmax 输出，PV MMA 是 f16f16f32，**P 必须保持 16-bit**）；
  - 新增 FP8-only 的 BF16 staging（单份、跨 stage 共享，BF16 路径用 `std::conditional_t`
    塌缩为 1 元素）：`ckv_bf16[CTA_TILE_KV * HEAD_DIM_CKV]`、`kpe_bf16[CTA_TILE_KV * HEAD_DIM_KPE]`。

### Day 2：`load_kv()` dtype-aware 化

- CKV 内层循环 `NUM_MMA_D_CKV/4` → `INNER_LOADS_CKV`（FP8 下 8→4，否则超界 2 倍）；
- KPE 循环换 `INNER_LOADS_KPE` 并加 lane gating：`if (lane_idx % 8 < LANES_PER_ROW_KPE)`
  （FP8 KPE 一行只有 4 个 b128，8 个 lane 会写穿）；
- KPE 写偏移换 `SWIZZLE_MODE_KPE_RAW`；
- 自检：gmem 指针步进 `8 * upcast_size<DTypeKV>()` 与新循环边界的乘积应恰好等于 head_dim。

### Day 3：新增 `repack_fp8_kv_to_bf16()`

照搬 `prefill.cuh:1016` / `mla_hopper.cuh` 的 idiom，SM80 版全员（256 线程）参与：

```cpp
// 每线程：读 16B 原始 FP8（16 个元素）→ vec_cast<DTypeQ, DTypeKV>::cast<16> →
// __hmul2 乘 variant.ckv_scale / kpe_scale → 写两个 16B 到 staging（k128B swizzle，
// 与 BF16 路径布局一致，QK/PV 的 ldmatrix 逻辑完全复用）
// 读侧：CKV 用 SWIZZLE_MODE_CKV，KPE 用 SWIZZLE_MODE_KPE_RAW（与 load_kv 写侧一致）
```

> **后续演进**：Day 9 的 scale 折叠优化把这里的两个 `__hmul2` 从 repack 热循环整体挪走
> （scale 折进 Q + fp32 epilogue），repack 退化为纯 `vec_cast`——见 Day 9 与 §0.6.9。

### Day 4：`compute_mla_qk()` / `compute_mla_pv()` 接 staging + P 的 DTypeQ 化

- FP8 路径下 `ckv_smem` / `kpe_smem` 指向 staging buffer，stride 用 `*_BF16` 版本
  （BF16 路径完全不变）；
- `compute_mla_pv()`：`p_f16` 数组、`vec_cast<., float>`、`m16k16_rowsum_f16f16f32`、
  MMA 模板参数从 `DTypeKV` 统一改为 `DTypeQ`（对齐 Hopper PR "P is always DTypeQ-typed"；
  BF16 路径下 DTypeQ==DTypeKV，行为逐 bit 不变）；
- V（=ckv）的 `ldmatrix_m8n8x4_trans` 读 staging，stride 用 `UPCAST_STRIDE_CKV_BF16`。

### Day 5：主循环同步 + smem dispatch + 编译通过

- 三处 compute 调用点（mask 循环 / no-mask 循环 / last-tiles 循环）在
  `wait_group + __syncthreads()` 之后、`compute_mla_qk` 之前插入
  `repack_fp8_kv_to_bf16(); __syncthreads();`（`if constexpr (USE_KV_REPACK)`）；
- **last-tiles 循环没有迭代间 barrier，必须在 repack 前额外补一个 `__syncthreads()`**
  （否则下一 tile 的 repack 会覆盖上一 tile PV 还在读的 staging）——这是 SM80 版特有的点；
- 同步安全性论证（写进代码注释）：
  - mask/no-mask 循环：下一 tile 的 repack 在下一轮迭代顶部 `__syncthreads()` 之后，
    彼时本轮 PV 对 staging 的读取已全部完成，单份 staging 安全；
  - prefetch（cp.async）写的是原始 FP8 stage buffer，repack 早已消费完毕，不冲突；
  - `wait_group<1>` 在飞的那组 cp.async 目标是另一个 stage buffer，与 repack 读的不同。
- `DISPATCH_SMEM_CONFIG` 增加 `kv_is_fp8` 分支阈值（见 0.5 的表），FP8 + 最低档
  直接 `FLASHINFER_ERROR` + `cudaErrorNotSupported`；
- 目标：`FLASHINFER_CUDA_ARCH_LIST=8.0` 下 JIT 编译通过（先不管对错）。

### Day 6–7：Python guard 放开 + 首次跑通

- `flashinfer/mla/_core.py::plan()` FP8 分支：从"仅 fa3 + SM90"放宽为
  "fa3 + SM90，**或 fa2 + SM80/SM90**"；sm86/89 继续拒绝且错误信息说明 smem 不足；
- 保留 q 必须 bf16、head_dim 必须 512/64 的限制（与 SM90 一致）；`run()` 无需改动；
- 在 A100 上手跑一个最小 case（batch=1, page_size=1, 短 kv），与 BF16 reference 对比。

### Day 8（延伸，实际追加）：prefetch 与 QK/PV 重叠的流水优化

> 这一步不在最初计划里，是 benchmark 发现 FP8 回退后尝试的性能优化，作为**真实开发经历**记录。

- **动机**：初版把 repack 做成独立 pass（`wait → __sync → repack → __sync → compute`），
  repack 是纯额外开销。观察到 **BF16 staging 让 compute 与原始 FP8 buffer 解耦**——repack 一完成，
  FP8 stage buffer 就可复用——于是把下一 tile 的 `cp.async` 预取**从 compute 尾部提前到 repack 之后**，
  让 DMA 覆盖整个 QK/PV 段，并省掉一个 `__syncthreads`（BF16 路径经 `if constexpr` 保持原样）。
- **踩坑（竞态）**：`kpe_p_smem` 是 `union{ kpe; p }`，QK_SHARD 下 `compute_mla_pv` 用 `.p` 做 P 的
  all-gather 中转。提前 prefetch 后，`load_kv` 的异步 cp.async 写 `.kpe` 与同轮 `compute_mla_pv`
  写 `.p`（union 同址）**数据竞争**，冲坏下一 tile 的 KPE → 下轮 repack 读到垃圾（输出 `1e32`）。
  `num_heads=16` 因 timing 没触发、`num_heads=128` 稳定复现——典型的竞态特征。
- **修复（零额外 smem）**：把 FP8 路径的 P all-gather 中转从 `.p` 改用 `kpe_bf16_smem`
  （QK 阶段已消费完、PV 阶段空闲，且大小正好 `HEAD_DIM_KPE == CTA_TILE_Q == 64` 相等），
  与 prefetch 写的 `.kpe` 彻底解耦。
- **结果**：加速比从 0.83–0.94x 抬到 **0.86–0.96x**（延迟降 ~3–4%），但**仍 < 1.0x**——
  因为 MLA compute-bound，能重叠的只是 DMA 延迟，repack 的 ALU/smem 指令本身是硬开销，
  无法像 SM90(warp specialization) 那样和 WGMMA 真正并行掩盖（见 §0.6）。
- 对应 commit：`perf: overlap FP8 KV prefetch with QK/PV on SM80 MLA`。

### Day 9（延伸，实际追加）：scale 折叠——把反量化 scale 挪出 repack 热循环

> 消融拆解（§0.6.9）显示 repack 的 dequant ALU 占 kernel ~5.9%，其中每 tile 32×576 个元素的
> `__hmul2` scale 乘法可被代数变换整体消掉。

- **数学**：`logits = q_nope·(s_c·ckv) + q_pe·(s_k·kpe) = (s_c·q_nope)·ckv + (s_k·q_pe)·kpe`，
  scale 折进 Q 后每个 **Q tile 只乘一次**（摊薄到全部 KV tile）；PV 侧
  `o = P·(s_c·V) = s_c·(P·V_raw)`，`ckv_scale` 在 epilogue 的 **fp32 o_frag** 上乘一次
  （精度反而优于 bf16 域乘法；P 是归一化概率，m/d/LSE 均不受影响）；
- **工程点**：Q 的 cp.async 单独 `commit_group()`，用带分支的 `wait_group<N>` 精确等
  "只剩 KV prologue 组在飞"——Q 到位即缩放，KV 预取流水不中断；
- **踩坑**：给 `KernelTraits` 加 `static_assert(NUM_STAGES == 2)` 断得太死——tier3
  （NUM_STAGES=1）虽然运行时不会给 FP8 派发，但模板仍会为 FP8 dtype **实例化**
  （dispatch 宏是运行时 if），须放宽为 `<= 2`；
- **结果**：fp8 kernel 再快 ~2%，加速比 0.86x → **0.87–0.88x**；162 FP8 + 6126 BF16 全过。
  对应 commit：`perf: fold FP8 dequant scales into Q and fp32 epilogue on SM80 MLA`（950c904d）。

### Day 10（延伸，实验，已回退）：寄存器路径 dequant——一次完整的负结果

> 消融地板（去掉 repack 后 fp8 比 bf16 快 5.7%）诱使我们尝试消灭 staging：
> QK 对 raw fp8 做 u16 视角 `ldmatrix` + 寄存器内 `fast_dequant`（Q 侧预做 σ 维度置换，
> 利用**点积对收缩维置换不变**；K 侧每 lane 拿到 4 个连续 dim 恰好构成 σ 下的两个 B 寄存器）；
> PV 对 raw fp8 做 u16 `ldmatrix.trans` + PRMT 奇偶抽列 + 偶/奇双 mma，epilogue 用
> width-4 shuffle 恢复 o_frag 布局。**功能一次写对（162 FP8 + 6126 BF16 全过），
> 但性能 0.73–0.78x < staging 的 0.87x，已回退**——根因、消融数据与双缓冲/Marlin 的
> 系统分析见 §0.6.9；实验代码存档 `/tmp/fp8_register_dequant_experiment.patch`。
> 教训：**动手前先用消融把"税"拆到管线粒度，并区分"消掉工作"与"搬动工作"**——
> 地板只对前者成立。

---

## 4. Week 4 — 测试、调试、性能与提交

### Day 1–2：正确性测试（**实际已完成，全过**）

- `tests/attention/test_deepseek_mla.py`：把 PR #3694 加的
  `test_batch_mla_fp8_kv_matches_bf16_reference` 等 FP8 测试**参数化 backend（fa2/fa3）**，
  SM80 设备上跑 fa2 分支，误差阈值对齐 SM90 标准；新增 `_skip_if_fp8_mla_unsupported()`
  按 (backend, device) 跳过；
- 覆盖维度：batch size、kv_len（跨 mask/no-mask/last-tiles 三个循环段的长短组合）、
  page_size（16 / 64）、causal 开关、num_heads（16 / 128）；
- **实测结果**：A100 上 **155 个 FP8 精度用例全过**；**BF16 回归 1350 用例零回归**
  （证实 P→DTypeQ、staging 改动对 BF16 路径逐 bit 无影响）。

### Day 3–4：调试与排错（预留缓冲，**实际用于定位流水优化的竞态**）

常用手段：
- [`compute-sanitizer`](https://docs.nvidia.com/compute-sanitizer/) 的
  `--tool memcheck / racecheck`（抓 smem 越界与竞态——swizzle 改错、循环边界改错、
  barrier 缺失基本都能抓到）；
- `FLASHINFER_JIT_DEBUG=1`（-O0 + 调试符号）+ `FLASHINFER_JIT_VERBOSE=1`
  （用法见 `CLAUDE.md` 与 `.claude/skills/debug-cuda-crash/skill.md`）；
- 分段验证法：先把 scale 固定为 1.0、KV 数值构造成 fp8 可精确表示的值（如小整数），
  此时 FP8 路径应与 BF16 路径**逐 bit 一致**，二分定位是 QK 段还是 PV 段出错；
- **实际经历**：流水优化引入的 union 竞态就是靠"`num_heads=128` 稳定复现 + `1e32` 垃圾值 +
  推断 `kpe_p_smem` union 同址"定位的（详见 Week 3 Day 8），也可用 racecheck 直接抓。

### Day 5：性能验证（**实际结论与最初假设相反**）

- benchmark 脚本：`benchmarks/bench_deepseek_mla_fp8_kv.py`（FP8 vs BF16 KV 的 MLA decode），
  对照脚本另测 GQA decode；用官方 `bench_gpu_time`；
- **最初假设（已证伪）**：以为"长 kv_len 下 FP8 省带宽 → 净收益为正"。**实测 FP8 反而回退 4–14%**
  （0.86–0.96x），因为 **MLA decode 是 compute-bound**（BF16 带宽只 ~340 GB/s，占 A100 HBM 峰值 ~17%），
  FP8 省的带宽兑不了现，多出的 dequant 是硬开销——**完整分析与数据见 §0.6**
  （后经 Day 9 的 scale 折叠收窄至 0.87–0.88x，且 §0.6.9 消融表明已近结构性上限）；
- **对照实验**（关键佐证）：同机测 GQA decode（memory-bound），FP8 **加速 1.2–1.6x**、
  BF16 带宽逼近 HBM 峰值 89%，反证 MLA 的 compute-bound 特性；
- **长序列不变性验证**：把 seq_len 扩到 32768/65536，加速比仍钉在 **0.86x**（§0.6.3），
  实测证伪"长序列更 memory-bound → FP8 变有效"的直觉——AI 与 seq_len 无关；
- **固定预算吞吐实验（已完成，§0.6.6）**：BF16 batch B vs FP8 batch 2B（等显存占用），
  吞吐比仅 **~0.85x（0.68–1.09x），无 2x**——因 compute 早已饱和，"容量翻倍"换不到吞吐；
- **结论**：FP8 对 MLA 的价值是**显存减半带来的可行性/容量**（能跑更长/更大、省卡），
  **既非 decode 提速、也非系统吞吐翻倍**。

### Day 6–7：收尾与提交

- `pre-commit run -a`（格式与 lint）；
- 整理 PR 描述（对照 PR #3694 的格式）：动机、设计（附 smem 预算表）、支持范围
  （SM80 ✅ / sm86,89 ❌ 及原因）、**精度数据 + 诚实的性能特征说明（compute-bound、FP8 主打省显存，
  附 §0.6 的 roofline 分析与 GQA 对照）**；
- 提交 PR 到 `flashinfer-ai/flashinfer`，根据 review 意见迭代。

---

## 5. 风险与应对

| 风险 | 影响 | 应对 | 实际结果 |
|---|---|---|---|
| KPE FP8 行宽 64B 触发 k128B swizzle 地址冲突 | 数据损坏、结果错误 | 照搬 Hopper PR 的 `SWIZZLE_MODE_KPE_RAW = k64B` 方案 | ✅ 已按方案实现，`test_fp8_kv_kpe_dominant_no_row_aliasing` 覆盖 |
| last-tiles 循环缺 barrier，staging 被提前覆盖 | 偶发精度错误（难复现） | 显式补 `__syncthreads()`；racecheck 验证 | ✅ 已补 |
| P 复用 `kpe_p_smem` 时 FP8 下空间不足 | 越界写 | union 按 DTypeQ 撑大 | ✅ union 已按 DTypeQ 撑大 |
| **（新）流水优化中 prefetch 的 cp.async 与 P all-gather 共用 union 竞态** | 静默错误输出（`1e32`），`num_heads=128` 复现 | P 中转改用空闲的 `kpe_bf16_smem` | ✅ 已修（Week 3 Day 8） |
| smem 预算算错导致 launch 失败 | `cudaFuncSetAttribute` 报错 | 以 `sizeof(SharedStorage)` 实测值核对表格 | ✅ nvcc static_assert 核验：152064/229888 精确吻合 |
| **（新）误以为 FP8 会给 MLA decode 提速** | 目标设错、性能承诺落空 | 用 roofline 判据 + 实测校正预期 | ⚠️ 已证伪：MLA compute-bound，FP8 不提速，价值在省显存（§0.6） |
| BF16 路径回归 | 存量用户受影响 | P→DTypeQ 等共路改动逐一论证"BF16 下数值不变"，全量回归兜底 | ✅ 1350 BF16 用例零回归 |
| **（新）性能优化方向选错，白费工程量** | 投入大改造（如寄存器路径 dequant）却负收益 | **先消融拆解定位"税"的构成，再动手；改造以可回退方式进行** | ⚠️ 寄存器路径实验负结果（0.74x），因有消融数据与 patch 存档，半天内干净回退（§0.6.9、Week 3 Day 10） |

## 6. 验收标准（附实际达成情况）

1. ✅ A100（SM80）上 `kv_data_type=torch.float8_e4m3fn` + `backend="fa2"` 的
   `BatchMLAPagedAttentionWrapper` 端到端跑通；
2. ✅ FP8 输出 vs BF16 reference 的误差满足与 SM90 相同的阈值（155 用例全过）；
3. ✅ 全量既有 BF16/FP16 MLA 测试零回归（1350 用例）；
4. ✅ sm86/sm89 在 Python(`plan()`) 与 C++(`DISPATCH_SMEM_CONFIG`) 两层都给出清晰报错，
   而非静默错误结果；
5. ⚠️ **原标准"长序列 decode FP8 有正吞吐收益"已按实测改写**：MLA decode 是 compute-bound，
   FP8 **无 decode 提速**（0.86–0.96x）。修正后的验收目标是——
   **(a)** 提供 roofline 分析 + GQA/MLA 对照实测，讲清楚"为什么不加速"（§0.6，已完成）；
   **(b)** FP8 **KV cache 显存精确减半**已由结构验证（每 token 1152B→576B）；
   **(c)** 固定显存预算下 FP8 batch 2B vs BF16 batch B 的吞吐实验**已完成**（§0.6.6）：
   吞吐 ~0.85x、无 2x，进一步确认"容量收益 ≠ 吞吐收益"；
6. ⏳ PR 提交至 upstream 并通过 CI（分支 `dev-plan-sm80-mla-fp8` 已就绪，待提 PR）。

## 7. 参考资料总表

论文：
- [Online normalizer calculation for softmax (arXiv:1805.02867)](https://arxiv.org/abs/1805.02867)
- [FlashAttention (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135) /
  [FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691)
- [FP8 Formats for Deep Learning (arXiv:2209.05433)](https://arxiv.org/abs/2209.05433)
- [DeepSeek-V2 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) /
  [DeepSeek-V3 (arXiv:2412.19437)](https://arxiv.org/pdf/2412.19437)（MLA、head 数）
- [vLLM / PagedAttention (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
- [FlashInfer (arXiv:2501.01005)](https://arxiv.org/abs/2501.01005)
- [Roofline (Williams et al., CACM 2009)](https://dl.acm.org/doi/10.1145/1498765.1498785)
  （§0.6 判断 FP8 是否加速的模型基础）
- [MARLIN: Mixed-Precision Auto-Regressive Parallel Inference (arXiv:2408.11743)](https://arxiv.org/abs/2408.11743)
  （fast dequantization 与 mainloop 融合的原始场景；对照 §0.6.9 理解其前提在 MLA 下不成立）

模型架构 / MLA 计算方式（§0.6 用）：
- [DeepSeek-V2-Lite · HF](https://huggingface.co/deepseek-ai/DeepSeek-V2-Lite)（16 heads）、
  GLM-5/5.1 `Glm5MoeDsa`（MLA，~64）、GLM-4.5/4.6 `glm4_moe`（GQA，非 MLA）
- sglang MLA 实现（absorbed decode vs un-absorbed `forward_mha.py` prefill）：
  `python/sglang/srt/models/deepseek_common/attention_forward_methods/`

官方文档：
- [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/)
- [PTX ISA](https://docs.nvidia.com/cuda/parallel-thread-execution/)（URL 加 `.md` 得 Markdown 版）
- [CUDA Math API（FP8）](https://docs.nvidia.com/cuda/cuda-math-api/)
- [Ampere Tuning Guide](https://docs.nvidia.com/cuda/ampere-tuning-guide/)
- [compute-sanitizer](https://docs.nvidia.com/compute-sanitizer/)
- [FlashInfer docs](https://docs.flashinfer.ai/) / [TVM-FFI](https://tvm.apache.org/ffi/)

书与课程：
- 《Programming Massively Parallel Processors》4th ed.
- 《C++ Templates: The Complete Guide》2nd ed.（工具书）
- [learncpp.com](https://www.learncpp.com/) / [cppreference.com](https://en.cppreference.com/)
- [GPU-MODE lectures](https://github.com/gpu-mode/lectures)

代码（最重要的教材）：
- [PR #3694](https://github.com/flashinfer-ai/flashinfer/pull/3694)（本计划的蓝本）
- `include/flashinfer/attention/mla_hopper.cuh`（SM90 实现，含设计理由注释）
- `include/flashinfer/attention/mla.cuh`（本次要改的 SM80 kernel）
- `include/flashinfer/attention/prefill.cuh::repack_fp8_tile_to_bf16`（SM80 FP8 idiom）
- `include/flashinfer/permuted_smem.cuh` / `include/flashinfer/mma.cuh` /
  `include/flashinfer/vec_dtypes.cuh`（重点 `fast_dequant_f8f16x4`）
- 本仓库 `CLAUDE.md` 与 `.claude/skills/`（JIT 开发流程、benchmark、调试教程）
- **本项目产出**：`benchmarks/bench_deepseek_mla_fp8_kv.py`（FP8/BF16 MLA decode 对比，§0.6 数据来源）；
  分支 `dev-plan-sm80-mla-fp8` 的 commit 序列（实现 → 测试 → 流水优化 → scale 折叠 →
  消融拆解与负结果归档）；`/tmp/fp8_register_dequant_experiment.patch`
  （寄存器路径 dequant 完整实现存档，含 σ 置换 / PRMT 抽列 / shuffle 重排的推导注释）
