#!/usr/bin/env python3
"""
Investigate where a fused RMSNorm+INT8-quant (Triton) kernel could plug into the INSTALLED vLLM.
Greps the installed vllm package for the relevant hooks and writes results/integration_map.md (file:line hits),
and reports which custom ops exist in this build. No GPU needed except the torch.ops check.
Usage: python -m vllm_eval.integration.inspect_integration
"""
import importlib.util, os, re, sys
from pathlib import Path
from vllm_eval import common as C

PATTERNS = [
    ("RMSNorm layer (dispatch to CUDA op / native)", r"class RMSNorm\b|def forward_cuda|fused_add_rms_norm|ops\.rms_norm\("),
    ("INT8 activation quant op (W8A8)", r"scaled_int8_quant"),
    ("Existing fused CUDA kernel binding", r"rms_norm_dynamic_per_token_quant"),
    ("W8A8 INT8 linear scheme (compressed-tensors)", r"CompressedTensorsW8A8Int8"),
    ("INT8 scaled-mm kernels (where quant is called)", r"CutlassScaledMMLinearKernel|Int8ScaledMM|class .*ScaledMM"),
    ("Qwen3 decoder layer (norm -> linear call sites)", r"class Qwen3DecoderLayer|input_layernorm|post_attention_layernorm"),
    ("Fusion passes (torch.compile pattern matcher)", r"RMSNormQuantFusionPass|class .*FusionPass|QuantKey|kInt8|rms_quant"),
    ("Custom-op registration (for a new Triton op)", r"direct_register_custom_op"),
    ("Out-of-tree plugin entry points", r"vllm\.general_plugins|load_general_plugins|register_oot"),
]


def main():
    spec = importlib.util.find_spec("vllm")
    if spec is None or not spec.submodule_search_locations:
        sys.exit("vllm is not installed in this environment")
    root = Path(list(spec.submodule_search_locations)[0])
    print(f"vllm package: {root}")
    files = [p for p in root.rglob("*.py") if "/tests/" not in str(p)]
    md = [f"# vLLM integration map (installed package: `{root}`)\n"]
    try:
        import vllm; md.append(f"vLLM version: **{vllm.__version__}**\n")
    except Exception: pass

    comp = root / "compilation"
    if comp.exists():
        md.append("## `vllm/compilation/` modules\n" + ", ".join(f"`{p.name}`" for p in sorted(comp.glob("*.py"))) + "\n")

    try:
        import torch, vllm._custom_ops  # noqa: F401  (loads the compiled _C extension)
        names = ["rms_norm", "fused_add_rms_norm", "scaled_int8_quant", "dynamic_scaled_int8_quant",
                 "static_scaled_int8_quant", "rms_norm_dynamic_per_token_quant", "cutlass_scaled_mm"]
        md.append("## Ops present in `torch.ops._C`\n" + "\n".join(f"- `{n}`: {hasattr(torch.ops._C, n)}" for n in names) + "\n")
    except Exception as e:
        md.append(f"## Ops present in `torch.ops._C`\ncould not check ({e})\n")

    for title, pat in PATTERNS:
        rx = re.compile(pat)
        hits = []
        for p in files:
            try:
                for i, line in enumerate(p.read_text(errors="ignore").splitlines(), 1):
                    if rx.search(line):
                        hits.append(f"- `{p.relative_to(root.parent)}:{i}`  `{line.strip()[:110]}`")
            except Exception:
                pass
        md.append(f"## {title}\npattern: `{pat}` ({len(hits)} hits)\n" + "\n".join(hits[:40]) + ("\n- ..." if len(hits) > 40 else "") + "\n")
        print(f"[{len(hits):3d} hits] {title}")
    C.RESULTS.mkdir(parents=True, exist_ok=True)
    (C.RESULTS / "integration_map.md").write_text("\n".join(md))
    print(f"[saved] {C.RESULTS / 'integration_map.md'}  (read it together with INTEGRATION_PROPOSAL.md)")


if __name__ == "__main__":
    main()
