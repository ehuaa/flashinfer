# FlashInfer SM80 (A100) MLA FP8 KV Cache 支持 — 4 周开发计划

> 目标：仿照 [PR #3694](https://github.com/flashinfer-ai/flashinfer/pull/3694)（SM90/Hopper MLA FP8 KV cache）的思路，
> 为 FA2 backend 的 MLA kernel（`include/flashinfer/attention/mla.cuh`）添加 SM80（A100）上的
> FP8 (e4m3) KV cache 支持。
>
> 计划总时长 4 周：**前 2 周为 CUDA / C++ / FlashInfer 学习**（面向零基础），**后 2 周为开发、测试与提交 PR**。

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
  `__nv_fp8_e4m3`、`vec_cast` 批量类型转换；
- 关键认知：**SM80 无 FP8 MMA，FP8 KV 的收益 = 显存减半 + gmem 带宽减半，代价 = 计算前反量化**。

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
- 本仓库 FP8 转换实现：`include/flashinfer/vec_dtypes.cuh` 的 `vec_cast<bf16, fp8_e4m3>`。

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

### Day 3：MLA 算法与 paged KV cache

学习内容：
- MLA（Multi-head Latent Attention）：ckv（512 维 compressed KV）+ kpe（64 维 rope K）、
  q_nope / q_pe 的双段 QK、为什么 ckv 既是 K 又是 V；
- paged KV cache：page_table / kv_indices / kv_indptr 的寻址方式；
- 对照 `load_kv()` 里 `block_size.divmod(packed_block_iter, q, r)` 理解 page 寻址。

参考资料：
- [DeepSeek-V2 论文 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434) 第 2.1 节
  （MLA 定义，重点看矩阵吸收后 attention 只作用于 c^KV 和 k^R 的形式——这正是
  kernel 里 ckv/kpe 两个 cache 的来源）；
- [vLLM / PagedAttention 论文 (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
  （paged KV cache 的动机与结构）；
- [FlashInfer 官方文档](https://docs.flashinfer.ai/) 的 MLA API 页
  （`BatchMLAPagedAttentionWrapper` 的参数语义：qo_indptr / kv_indptr / kv_indices / kv_len_arr）；
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
- `tests/attention/test_deepseek_mla.py` 的 FP8 测试段（量化 helper、误差阈值的设定依据）。

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

---

## 4. Week 4 — 测试、调试、性能与提交

### Day 1–2：正确性测试

- `tests/attention/test_deepseek_mla.py`：把 PR #3694 加的
  `test_batch_mla_fp8_kv_matches_bf16_reference` 等 FP8 测试**参数化 backend（fa2/fa3）**，
  SM80 设备上跑 fa2 分支，误差阈值对齐 SM90 标准；
- 覆盖维度：batch size、kv_len（跨 mask/no-mask/last-tiles 三个循环段的长短组合）、
  page_size（1 / 64）、causal 开关、num_heads（16/64/128，触发不同 split 调度）；
- **BF16 回归**：跑全量既有 MLA 测试，确认 P→DTypeQ 等改动对 BF16 路径零影响。

### Day 3–4：调试与排错（预留缓冲）

常用手段：
- [`compute-sanitizer`](https://docs.nvidia.com/compute-sanitizer/) 的
  `--tool memcheck / racecheck`（抓 smem 越界与竞态——swizzle 改错、循环边界改错、
  barrier 缺失基本都能抓到）；
- `FLASHINFER_JIT_DEBUG=1`（-O0 + 调试符号）+ `FLASHINFER_JIT_VERBOSE=1`
  （用法见 `CLAUDE.md` 与 `.claude/skills/debug-cuda-crash/skill.md`）；
- 分段验证法：先把 scale 固定为 1.0、KV 数值构造成 fp8 可精确表示的值（如小整数），
  此时 FP8 路径应与 BF16 路径**逐 bit 一致**，二分定位是 QK 段还是 PV 段出错。

### Day 5：性能验证

- `benchmarks/flashinfer_benchmark.py` 对比 FP8 vs BF16 KV 的 decode 吞吐
  （用法见 `.claude/skills/benchmark-kernel/skill.md`；关注长 kv_len：FP8 的收益是
  KV 显存减半 + gmem 带宽减半，A100 上 repack 吃一点 SM 算力，预期长序列下净收益为正）；
- 若 repack 成为瓶颈，可尝试的优化（记录数据后再决定）：repack 与 cp.async 重叠、
  只对 ckv 做 staging 而 kpe 走寄存器 dequant。

### Day 6–7：收尾与提交

- `pre-commit run -a`（格式与 lint）；
- 整理 PR 描述（对照 PR #3694 的格式）：动机、设计（附 smem 预算表）、支持范围
  （SM80 ✅ / sm86,89 ❌ 及原因）、精度与性能数据；
- 提交 PR 到 `flashinfer-ai/flashinfer`，根据 review 意见迭代。

---

## 5. 风险与应对

| 风险 | 影响 | 应对 |
|---|---|---|
| KPE FP8 行宽 64B 触发 k128B swizzle 地址冲突 | 数据损坏、结果错误 | 照搬 Hopper PR 的 `SWIZZLE_MODE_KPE_RAW = k64B` 方案 |
| last-tiles 循环缺 barrier，staging 被提前覆盖 | 偶发精度错误（难复现） | Week 3 Day 5 显式补 `__syncthreads()`；racecheck 验证 |
| P 复用 `kpe_p_smem` 时 FP8 下空间不足 | 越界写 | union 按 DTypeQ 撑大；0.5 的预算表已含此项 |
| smem 预算算错导致 launch 失败 | `cudaFuncSetAttribute` 报错 | 以 `sizeof(SharedStorage)` 实测值核对表格；A100 实测 |
| 学习周进度不及预期 | 挤压开发时间 | Week 1/2 的验收标准严格执行；Week 4 Day 3–4 调试日可作缓冲 |
| BF16 路径回归 | 存量用户受影响 | P→DTypeQ 等共路改动逐一论证"BF16 下数值不变"，并靠全量回归测试兜底 |

## 6. 验收标准

1. A100（SM80）上 `kv_data_type=torch.float8_e4m3fn` + `backend="fa2"` 的
   `BatchMLAPagedAttentionWrapper` 端到端跑通；
2. FP8 输出 vs BF16 reference 的误差满足与 SM90 相同的阈值；scale=1 + 可精确表示数值时逐 bit 一致；
3. 全量既有 BF16/FP16 MLA 测试零回归；
4. sm86/sm89 上给出清晰报错而非静默错误结果；
5. 长序列 decode 场景 FP8 相对 BF16 有正的吞吐收益（附 benchmark 数据）；
6. PR 提交至 upstream 并通过 CI。

## 7. 参考资料总表

论文：
- [Online normalizer calculation for softmax (arXiv:1805.02867)](https://arxiv.org/abs/1805.02867)
- [FlashAttention (arXiv:2205.14135)](https://arxiv.org/abs/2205.14135) /
  [FlashAttention-2 (arXiv:2307.08691)](https://arxiv.org/abs/2307.08691)
- [FP8 Formats for Deep Learning (arXiv:2209.05433)](https://arxiv.org/abs/2209.05433)
- [DeepSeek-V2 (arXiv:2405.04434)](https://arxiv.org/abs/2405.04434)
- [vLLM / PagedAttention (arXiv:2309.06180)](https://arxiv.org/abs/2309.06180)
- [FlashInfer (arXiv:2501.01005)](https://arxiv.org/abs/2501.01005)

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
- `include/flashinfer/permuted_smem.cuh` / `include/flashinfer/mma.cuh` / `include/flashinfer/vec_dtypes.cuh`
- 本仓库 `CLAUDE.md` 与 `.claude/skills/`（JIT 开发流程、benchmark、调试教程）
