#!/usr/bin/env python3
"""Offline tests (no GPU, no vLLM): kernel classification, fusable-region detection, trace parsing, subtraction, Amdahl."""
import json, math, tempfile, os
from vllm_eval import analysis as A

K = {
    "addnorm": "void vllm::fused_add_rms_norm_kernel<c10::BFloat16, 8>(c10::BFloat16*, c10::BFloat16*, c10::BFloat16 const*, float, int, int)",
    "norm": "void vllm::rms_norm_kernel<c10::BFloat16, 8>(c10::BFloat16*, c10::BFloat16 const*, c10::BFloat16 const*, float, int, int, int)",
    "quant": "void vllm::dynamic_scaled_int8_quant_kernel<c10::BFloat16, float, int>(c10::BFloat16 const*, signed char*, float*, int)",
    "gemm": "cutlass_80_tensorop_s8t_s8n_s32t_gemm_256x128",
    "rope": "void vllm::rotary_embedding_kernel<c10::BFloat16, true>(...)",
    "attn": "void flash_fwd_kernel<Flash_fwd_kernel_traits<128>>(...)",
    "act": "void vllm::act_and_mul_kernel<c10::BFloat16, &vllm::silu_kernel>(...)",
    "fusedcuda": "void vllm::rms_norm_dynamic_per_token_quant_kernel<c10::BFloat16, signed char, true>(...)",
    "inductor_norm": "triton_red_fused__to_copy_add_mean_mul_pow_rsqrt_0",
    "inductor_quant": "triton_per_fused__to_copy_abs_amax_clamp_div_round_1",
    "memcpy": "Memcpy DtoD",
    # names seen with vLLM 0.34 + torch.compile on the L4 (quant op stays an extern CUDA kernel)
    "ind_addnorm": "triton_red_fused_dynamic_scaled_int8_quant_fused_add_rms_norm_2",
    "ind_silu": "triton_poi_fused_dynamic_scaled_int8_quant_mul_silu_slice_1",
    "ind_qknorm_rope": "triton_per_fused_add_cat_index_select_mul_rms_norm_split_split_with_sizes_sub_unsqueeze_view_3",
}


def synth_layer(t0):
    """One decoder layer: [add-norm, quant, qkv gemm, q_norm, rope, attn, quant, o gemm, add-norm, quant, gate_up gemm, silu, quant, down gemm]."""
    seq = [("addnorm", 10), ("quant", 5), ("gemm", 100), ("norm", 3), ("rope", 4), ("attn", 50), ("quant", 5),
           ("gemm", 40), ("addnorm", 10), ("quant", 5), ("gemm", 200), ("act", 10), ("quant", 8), ("gemm", 100)]
    ev, t = [], t0
    for k, d in seq:
        ev.append({"ph": "X", "cat": "kernel", "name": K[k], "ts": t, "dur": d, "pid": 0, "tid": 7, "args": {"stream": 7}}); t += d + 1
    return ev, t


def main():
    # classification
    exp = {"addnorm": "norm", "norm": "norm", "quant": "quant", "gemm": "gemm", "rope": "rope", "attn": "attention",
           "act": "activation", "fusedcuda": "norm_quant_fused", "inductor_norm": "norm", "inductor_quant": "quant",
           "ind_addnorm": "norm", "ind_silu": "activation", "ind_qknorm_rope": "norm"}
    for k, c in exp.items():
        assert A.classify(K[k]) == c, (k, A.classify(K[k]), c)
    assert A.classify(K["memcpy"], "gpu_memcpy") == "memory"

    # trace parsing + analysis over 2 layers
    evs, t = [], 0
    for _ in range(2):
        e, t = synth_layer(t); evs += e
    path = os.path.join(tempfile.mkdtemp(), "trace.json")
    json.dump({"traceEvents": evs + [{"ph": "X", "cat": "cpu_op", "name": "ignored", "ts": 0, "dur": 999}]}, open(path, "w"))
    m = A.analyze_events(A.load_gpu_events(path))
    assert m["n_kernels"] == 28
    c = m["by_cat_us"]
    assert c["norm"] == 2 * 23 and c["quant"] == 2 * 23, c          # norms: 10+3+10, quants: 5+5+5+8
    assert m["fusable"]["quant_us"] == 2 * 10 and m["fusable"]["norm_us"] == 2 * 20, m["fusable"]  # only qkv & gate_up quant follow a norm
    assert m["fusable"]["quant_n"] == 4
    s = A.shares(m)
    assert abs(s["norm_plus_quant"] - 92 / m["total_us"]) < 1e-9

    # decode-only subtraction
    mb, ma = m, A.analyze_events(A.load_gpu_events(path)[:14])
    d = A.subtract(mb, ma)
    assert d["by_cat_us"]["quant"] == 23 and d["total_us"] == mb["total_us"] - ma["total_us"]

    # Amdahl
    assert abs(A.amdahl(0.1, 2) - 1 / (0.9 + 0.05)) < 1e-12
    assert abs(A.amdahl(0.1, math.inf) - 1 / 0.9) < 1e-12
    assert abs(A.S_BW_BOUND - 11 / 7) < 1e-12
    print("selftest OK: classification, fusable-region detection, trace parsing, subtraction, Amdahl")


if __name__ == "__main__":
    main()
