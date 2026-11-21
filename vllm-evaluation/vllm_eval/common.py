"""Shared helpers: environment guard (NVIDIA L4), prompt generation, vLLM factory, stats, JSON I/O."""
import os, sys, re, json, time, random, statistics, subprocess, platform
from pathlib import Path

# Run the vLLM engine core in-process so torch.profiler can see the model's CUDA kernels.
# Must be set before vllm is imported.
os.environ.setdefault("VLLM_ENABLE_V1_MULTIPROCESSING", "0")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")   # use the (single) L4

W8A8_MODEL = "nytopop/Qwen3-1.7B.w8a8"
BF16_MODEL = "Qwen/Qwen3-1.7B"
MODELS = {"w8a8": W8A8_MODEL, "bf16": BF16_MODEL}
ROOT = Path(__file__).resolve().parent.parent
RESULTS = Path(os.environ["RESULTS_DIR"]) if os.environ.get("RESULTS_DIR") else ROOT / "results"
SEED = 0

# Fixed workloads (from the task)
PREFILL_WORKLOADS = {"8x512": (8, 512), "1x2048": (1, 2048)}   # (num_prompts, prompt_len)
DECODE_BATCHES = [1, 8, 64]
DECODE_OUT_LEN = 128


def is_l4(name: str) -> bool:
    return re.search(r"\bL4\b", name) is not None      # matches "NVIDIA L4", not "L40S"


def require_l4(allow: bool = False):
    import torch
    if not torch.cuda.is_available():
        sys.exit("CUDA is not available. Run on the L4 machine inside the vLLM environment.")
    name = torch.cuda.get_device_name(0)
    if not is_l4(name) and not (allow or os.environ.get("ALLOW_NON_L4") == "1"):
        sys.exit(f"GPU is '{name}', not an NVIDIA L4. Use --allow-non-l4 (or ALLOW_NON_L4=1) to override.")
    return name


def smi_snapshot():
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=name,driver_version,clocks.sm,clocks.max.sm,clocks.mem,"
                              "power.draw,power.limit,temperature.gpu,memory.used,memory.total",
                              "--format=csv,noheader"], capture_output=True, text=True, timeout=20).stdout.strip()
        return out
    except Exception as e:
        return f"nvidia-smi unavailable: {e}"


def env_info():
    import torch
    info = {"python": platform.python_version(), "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": torch.cuda.get_device_name(0), "compute_capability": ".".join(map(str, torch.cuda.get_device_capability(0))),
            "smi": smi_snapshot()}
    try:
        import vllm; info["vllm"] = vllm.__version__
    except Exception as e:
        info["vllm"] = f"import failed: {e}"
    try:
        import transformers; info["transformers"] = transformers.__version__
    except Exception:
        pass
    return info


def make_prompts(n, length, seed=SEED, lo=1000, hi=30000):
    """n random token-id prompts of exact length (no shared prefixes)."""
    rng = random.Random(seed * 1000003 + n * 7919 + length)
    return [{"prompt_token_ids": [rng.randint(lo, hi) for _ in range(length)]} for _ in range(n)]


def make_llm(model, max_model_len=4096, gpu_mem=0.85, enforce_eager=False, custom_ops=None,
             max_num_seqs=64, max_num_batched_tokens=8192, seed=SEED, **extra):
    """Stock vLLM engine. Quantization is auto-detected from the checkpoint config (no quantization= override).
    Prefix caching is OFF so repeated identical prompts are really recomputed."""
    from vllm import LLM
    kw = dict(model=model, dtype="bfloat16", max_model_len=max_model_len, gpu_memory_utilization=gpu_mem,
              enable_prefix_caching=False, max_num_seqs=max_num_seqs,
              max_num_batched_tokens=max_num_batched_tokens,   # >= 8x512 so a prefill batch is not chunked
              seed=seed, enforce_eager=enforce_eager)
    if custom_ops:
        kw["compilation_config"] = {"custom_ops": [c.strip() for c in custom_ops.split(",") if c.strip()]}
    kw.update(extra)
    t0 = time.perf_counter()
    llm = LLM(**kw)
    print(f"[llm] loaded {model} in {time.perf_counter() - t0:.1f}s | kwargs={ {k: v for k, v in kw.items() if k != 'model'} }")
    return llm


def detected_quantization(llm):
    try:
        return str(llm.llm_engine.model_config.quantization)
    except Exception:
        return "unknown"


def sampling(max_tokens, **kw):
    from vllm import SamplingParams
    return SamplingParams(max_tokens=max_tokens, temperature=0.0, ignore_eos=True,
                          min_tokens=max_tokens if max_tokens > 1 else 0, **kw)


def timed_generate(llm, prompts, sp):
    t0 = time.perf_counter()
    outs = llm.generate(prompts, sp, use_tqdm=False)
    return time.perf_counter() - t0, outs


def summarize(xs):
    xs = list(xs)
    return {"median": statistics.median(xs), "mean": statistics.fmean(xs),
            "stdev": statistics.stdev(xs) if len(xs) > 1 else 0.0, "min": min(xs), "max": max(xs), "n": len(xs)}


def save_json(obj, name):
    RESULTS.mkdir(parents=True, exist_ok=True)
    p = RESULTS / name
    p.write_text(json.dumps(obj, indent=2, default=str))
    print(f"[saved] {p}")
    return p


def load_json(name):
    p = RESULTS / name
    return json.loads(p.read_text()) if p.exists() else None


def add_common_args(p):
    p.add_argument("--tag", choices=list(MODELS), required=True, help="w8a8 or bf16")
    p.add_argument("--model", default=None, help="override model id")
    p.add_argument("--max-model-len", type=int, default=4096)
    p.add_argument("--gpu-mem", type=float, default=0.85, help="vLLM gpu_memory_utilization (L4 has 24 GB)")
    p.add_argument("--enforce-eager", action="store_true", help="disable CUDA graphs / compile (clearer kernel attribution)")
    p.add_argument("--custom-ops", default=None, help='e.g. "+rms_norm" to force vLLM CUDA custom ops')
    p.add_argument("--allow-non-l4", action="store_true")
    return p
