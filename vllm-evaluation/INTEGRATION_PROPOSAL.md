# Proposed integration approach for V3 (fused Triton RMSNorm + INT8 quantization) in vLLM

> Status: **proposal**. Hook locations must be confirmed against the installed vLLM version with
> `python -m vllm_eval.integration.inspect_integration` (writes `results/integration_map.md` with file:line hits). Names below are the
> concepts to look for, not guaranteed paths.

## 1. What the W8A8 model does per decoder layer (Qwen3-1.7B)

```
residual-add + input_layernorm  ->  [quant]  -> qkv_proj (INT8 GEMM)  -> q_norm/k_norm -> RoPE -> attention
                                    [quant]  -> o_proj  (INT8 GEMM)
residual-add + post_attn_layernorm -> [quant] -> gate_up_proj (INT8 GEMM) -> SiLU*mul
                                    [quant]  -> down_proj (INT8 GEMM)
```
- 4 dynamic per-token INT8 quantizations per layer, but only **2 directly follow an RMSNorm** (qkv_proj and gate_up_proj).
  o_proj's input comes from attention and down_proj's from SiLU-and-mul, so they cannot be fused with a norm.
- q_norm / k_norm are RMSNorms followed by RoPE (no quantization): not fusable with quant.
- `profile_breakdown.py` measures exactly this: the **fusable region** is the norm kernel immediately followed by a
  quant kernel on the same stream. The Amdahl bound should use that fraction (`f_fusable`), not all norm+quant time.

## 2. Hooks in vLLM (look these up with `inspect_integration.py`)

| Concept | What to look for | Role |
|---|---|---|
| RMSNorm layer | `class RMSNorm`, `forward_cuda` (`ops.rms_norm`, `ops.fused_add_rms_norm`) | produces the normalized activation |
| INT8 activation quant | `ops.scaled_int8_quant` (dynamic per-token, symmetric, no zero-point) | called inside the W8A8 linear path |
| W8A8 linear scheme | `CompressedTensorsW8A8Int8` and the scaled-mm kernel (`CutlassScaledMMLinearKernel`) `apply_weights` | quantize -> `cutlass_scaled_mm` |
| Existing fused kernel | `rms_norm_dynamic_per_token_quant` binding in `_custom_ops` | exists, but INT8 models never use it |
| Compile fusion pass | `RMSNormQuantFusionPass` (fusion pass module under `vllm/compilation/`) | has FP8 / NVFP4 patterns, **no INT8** |
| Op registration | `direct_register_custom_op` | needed to expose a Triton op to torch.compile / CUDA graphs |
| Plugins | `vllm.general_plugins` entry point | patch vLLM without forking |

## 3. Integration options

**Option A: layer-level (manual) fusion via a plugin. Recommended for the first end-to-end numbers.**
1. Register a custom op `torch.ops.vllm_ws.rms_norm_int8(x, residual, weight, eps) -> (q_int8, scales_fp32[T,1])`
   (Triton V3 kernel, in-place residual update, with a fake/meta implementation).
2. Patch the decoder layer (or a generic "norm -> linear" pair) so that after `input_layernorm` and
   `post_attention_layernorm` the fused op is called instead of RMSNorm.
3. Teach the W8A8 INT8 linear path to accept pre-quantized input `(q, scales)` and skip its own `scaled_int8_quant`
   (patch `apply_weights`, or pass a small wrapper object / attribute carrying `(q, scales)` along with the tensor).
4. Package as a `vllm.general_plugins` entry point so stock vLLM runs unmodified on disk.
- Pros: independent of the compiler, easy to A/B (env flag), works with CUDA graphs. Cons: model-specific patching,
  must be redone per model family; upstream is moving toward similar manual fusion (see proposal ref. [18]).

**Option B: torch.compile fusion pass (the upstream-style path).**
Add INT8 patterns (`rms_norm`/`fused_add_rms_norm` followed by `scaled_int8_quant`) to the existing RMSNorm+quant
fusion pass and replace them with the V3 op. Requires the custom ops to be enabled (`custom_ops: ["+rms_norm", ...]`) so
the pattern matcher sees them, and the V3 op to be registered with `direct_register_custom_op`.
- Pros: model-agnostic, upstreamable. Cons: more engineering, depends on compile mode and pattern-matcher details.

**Option C: swap only the CUDA kernel for a Triton one.**
Not useful alone, because INT8 models do not call `rms_norm_dynamic_per_token_quant` today. Use only together with A or B.

**Recommendation:** prototype with **A** to measure real end-to-end gains against the Amdahl bound, then port to **B**
if the gain justifies an upstream PR.

## 4. Contract the V3 kernel must satisfy
- Inputs: `x [T,H]` BF16, `residual [T,H]` BF16 (updated in place: `residual <- x + residual`), `weight [H]`, `eps`.
- Outputs: `q [T,H]` INT8, `scales [T,1]` FP32 (dynamic per-token, symmetric: `scale = absmax/127`, no zero-point).
- Same rounding/clamping as vLLM's quant op (compare INT8 mismatch rate; the issue reports 0.2-1.1% off by <= 2 steps).
- CUDA-graph safe: no host sync, fixed grid per captured shape, **Triton autotune finished before graph capture**.
- Hidden size 2048 (Qwen3-1.7B) and 4096 (8B) must both be supported.

## 5. Risks and checks
| Risk | Check |
|---|---|
| Only 2 of 4 quant sites fusable | use `f_fusable` in Amdahl; confirm kernel count per layer drops by 2 in the new profile |
| Numerical drift | per-token scale and dequantized error vs FP32; WikiText-2 PPL unchanged (`eval_ppl.py`) |
| CUDA graph capture failure | run with graphs on and `--enforce-eager`; compare |
| Prefill regression (the known 2x slowdown of the CUDA kernel) | profile prefill 8x512 and decode b1/8/64 before and after |
| Compile/trace issues for the custom op | provide fake impl; test with and without `custom_ops` |

## 6. Validation plan
1. Unit test: V3 vs V0 (`fused_add_rms_norm` + `scaled_int8_quant`) on random data, both hidden sizes.
2. `eval_ppl.py` with the plugin enabled: PPL within noise of the W8A8 baseline.
3. `bench_perf.py` with the plugin on/off: prefill tok/s and decode TPOT.
4. `profile_breakdown.py` again: norm+quant share and kernel counts; compare achieved speedup with `amdahl.md`.

## 7. Sketch of the plugin (illustrative, untested)
```python
# vllm_ws_plugin/__init__.py   (entry point group: "vllm.general_plugins")
def register():
    import torch
    from vllm.utils import direct_register_custom_op        # location differs across vLLM versions
    from .triton_kernel import rms_norm_int8_triton          # V3 (Triton)

    def impl(x, residual, weight, eps):
        return rms_norm_int8_triton(x, residual, weight, eps)   # -> (q, scales), residual updated in place
    def fake(x, residual, weight, eps):
        return (torch.empty_like(x, dtype=torch.int8), x.new_empty((x.shape[0], 1), dtype=torch.float32))
    direct_register_custom_op("rms_norm_int8_triton", impl, mutates_args=["residual"], fake_impl=fake)
    # then wrap Qwen3DecoderLayer.forward and the W8A8 INT8 linear apply_weights (see integration_map.md for the exact symbols)
```
