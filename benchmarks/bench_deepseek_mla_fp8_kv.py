"""Benchmark FP8 vs BF16 KV cache for the FA2 MLA decode kernel (SM80/SM90).

FP8 KV halves the KV-cache footprint and global-memory traffic; on SM80 the
kernel dequantizes each tile to BF16 in shared memory before the MMA, so the
compute cost is unchanged and the win shows up on memory-bound (long-context /
large-batch) decode.
"""

import numpy as np
import torch

import flashinfer
from flashinfer.testing.utils import bench_gpu_time

HEAD_DIM_CKV = 512
HEAD_DIM_KPE = 64


def _per_tensor_symmetric_quant_fp8(x, fp8_max=448.0):
    amax = x.abs().max().item()
    scale = amax / fp8_max if amax > 0 else 1.0
    q = (x / scale).clamp(-fp8_max, fp8_max).to(torch.float8_e4m3fn)
    return q, scale


def _bench_one(batch_size, seq_len, num_heads, page_size, kv_dtype, backend):
    device = "cuda"
    q_nope = torch.randn(
        batch_size, num_heads, HEAD_DIM_CKV, dtype=torch.bfloat16, device=device
    )
    q_pe = torch.randn(
        batch_size, num_heads, HEAD_DIM_KPE, dtype=torch.bfloat16, device=device
    )

    num_pages = (batch_size * seq_len + page_size - 1) // page_size
    ckv_bf16 = torch.randn(num_pages, page_size, HEAD_DIM_CKV, device=device) * 0.1
    kpe_bf16 = torch.randn(num_pages, page_size, HEAD_DIM_KPE, device=device) * 0.1

    ckv_scale = kpe_scale = None
    if kv_dtype == torch.float8_e4m3fn:
        ckv, ckv_scale = _per_tensor_symmetric_quant_fp8(ckv_bf16)
        kpe, kpe_scale = _per_tensor_symmetric_quant_fp8(kpe_bf16)
    else:
        ckv = ckv_bf16.to(kv_dtype)
        kpe = kpe_bf16.to(kv_dtype)

    sm_scale = 1.0 / ((HEAD_DIM_CKV + HEAD_DIM_KPE) ** 0.5)
    workspace = torch.empty(256 * 1024 * 1024, dtype=torch.int8, device=device)
    wrapper = flashinfer.mla.BatchMLAPagedAttentionWrapper(workspace, backend=backend)

    q_indptr = torch.arange(0, batch_size + 1, device=device).int()
    kv_indptr = torch.arange(0, batch_size + 1, device=device).int() * (
        seq_len // page_size
    )
    kv_indices = torch.arange(0, num_pages, device=device).int()
    kv_lens = torch.full((batch_size,), seq_len, dtype=torch.int32, device=device)

    wrapper.plan(
        q_indptr,
        kv_indptr,
        kv_indices,
        kv_lens,
        num_heads,
        HEAD_DIM_CKV,
        HEAD_DIM_KPE,
        page_size,
        False,  # causal
        sm_scale,
        q_nope.dtype,
        kv_dtype,
    )
    kwargs = {}
    if ckv_scale is not None:
        kwargs = {"ckv_scale": ckv_scale, "kpe_scale": kpe_scale}

    o = wrapper.run(q_nope, q_pe, ckv, kpe, **kwargs)
    measurements = bench_gpu_time(
        lambda: wrapper.run(q_nope, q_pe, ckv, kpe, **kwargs),
        dry_run_time_ms=100,
        repeat_time_ms=500,
    )
    ms = np.median(measurements)
    # KV-cache bytes dominate decode traffic (q/o are per-token, KV is per-seq).
    kv_bytes = ckv.numel() * ckv.element_size() + kpe.numel() * kpe.element_size()
    io = kv_bytes + sum(t.numel() * t.element_size() for t in (q_nope, q_pe, o))
    return ms, io


def fixed_budget_throughput(backend="fa2", page_size=64):
    """Fixed-HBM-budget throughput: BF16 at batch B vs FP8 at batch 2B (equal KV
    footprint). If MLA decode were memory-capacity-bound with spare compute, FP8
    would ~double throughput. Since SM80 MLA decode is compute-bound, throughput
    (requests-steps / s) is expected to stay ~flat (~0.86x), not 2x."""
    print("\n=== fixed HBM budget: BF16 batch B vs FP8 batch 2B (equal KV bytes) ===")
    header = (
        f"{'heads':>5} {'seq_len':>7} {'B(bf16)':>8} {'2B(fp8)':>8} | "
        f"{'bf16 ms':>8} {'fp8 ms':>8} | "
        f"{'bf16 kreq/s':>11} {'fp8 kreq/s':>11} {'thrpt x':>8}"
    )
    print(header)
    print("-" * len(header))
    for num_heads in [16, 128]:
        for seq_len in [16384, 65536]:
            for batch_b in [8, 16]:
                ms_bf16, _ = _bench_one(
                    batch_b, seq_len, num_heads, page_size, torch.bfloat16, backend
                )
                ms_fp8, _ = _bench_one(
                    2 * batch_b, seq_len, num_heads, page_size,
                    torch.float8_e4m3fn, backend,
                )
                # throughput = requests advanced one decode step per second
                tp_bf16 = batch_b / (ms_bf16 * 1e-3) / 1e3
                tp_fp8 = (2 * batch_b) / (ms_fp8 * 1e-3) / 1e3
                print(
                    f"{num_heads:>5} {seq_len:>7} {batch_b:>8} {2 * batch_b:>8} | "
                    f"{ms_bf16:>8.4f} {ms_fp8:>8.4f} | "
                    f"{tp_bf16:>11.2f} {tp_fp8:>11.2f} {tp_fp8 / tp_bf16:>7.2f}x"
                )


def main():
    backend = "fa2"
    page_size = 64
    print(f"backend={backend}, page_size={page_size}, head_dim=512+64, q=bf16\n")
    header = (
        f"{'batch':>5} {'seq_len':>7} {'heads':>5} | "
        f"{'bf16 ms':>9} {'fp8 ms':>9} {'speedup':>7} | "
        f"{'bf16 GB/s':>9} {'fp8 GB/s':>9}"
    )
    print(header)
    print("-" * len(header))
    for num_heads in [16, 128]:
        for seq_len in [1024, 4096, 16384, 32768, 65536]:
            for batch_size in [16, 64]:
                ms_bf16, io_bf16 = _bench_one(
                    batch_size, seq_len, num_heads, page_size, torch.bfloat16, backend
                )
                ms_fp8, io_fp8 = _bench_one(
                    batch_size,
                    seq_len,
                    num_heads,
                    page_size,
                    torch.float8_e4m3fn,
                    backend,
                )
                bw_bf16 = io_bf16 * 1e-6 / ms_bf16
                bw_fp8 = io_fp8 * 1e-6 / ms_fp8
                print(
                    f"{batch_size:>5} {seq_len:>7} {num_heads:>5} | "
                    f"{ms_bf16:>9.4f} {ms_fp8:>9.4f} {ms_bf16 / ms_fp8:>6.2f}x | "
                    f"{bw_bf16:>9.1f} {bw_fp8:>9.1f}"
                )


if __name__ == "__main__":
    main()
    fixed_budget_throughput()
