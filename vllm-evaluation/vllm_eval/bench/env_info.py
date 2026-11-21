#!/usr/bin/env python3
"""Record the environment (GPU, driver, torch, vLLM versions) to results/env.json."""
from vllm_eval import common
name = common.require_l4(allow=False)
info = common.env_info()
print(info)
common.save_json(info, "env.json")
