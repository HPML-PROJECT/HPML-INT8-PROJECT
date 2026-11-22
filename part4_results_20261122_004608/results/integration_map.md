# vLLM integration map (installed package: `/home/ta2828/rmsnorm_int8_bench/.venv/lib/python3.13/site-packages/vllm`)

vLLM version: **0.34.0**

## `vllm/compilation/` modules
`__init__.py`, `backends.py`, `base_static_graph.py`, `breakable_cudagraph.py`, `caching.py`, `codegen.py`, `compiler_interface.py`, `counter.py`, `cuda_graph.py`, `cudagraph_pool.py`, `decorators.py`, `depyf_compat.py`, `monitor.py`, `partition_rules.py`, `piecewise_backend.py`, `wrapper.py`

## Ops present in `torch.ops._C`
- `rms_norm`: True
- `fused_add_rms_norm`: True
- `scaled_int8_quant`: False
- `dynamic_scaled_int8_quant`: True
- `static_scaled_int8_quant`: True
- `rms_norm_dynamic_per_token_quant`: True
- `cutlass_scaled_mm`: True

## RMSNorm layer (dispatch to CUDA op / native)
pattern: `class RMSNorm\b|def forward_cuda|fused_add_rms_norm|ops\.rms_norm\(` (196 hits)
- `vllm/_aiter_ops.py:1169`  `return torch.ops.vllm_aiter.fused_add_rms_norm(`
- `vllm/_custom_ops.py:293`  `def fused_add_rms_norm(`
- `vllm/_custom_ops.py:300`  `torch.ops._C.fused_add_rms_norm(input, residual, weight, epsilon)`
- `vllm/config/kernel.py:34`  `fused_add_rms_norm: list[str] = Field(default_factory=list)`
- `vllm/config/kernel.py:35`  `"""Priority list for vllm.ir.ops.fused_add_rms_norm"""`
- `vllm/config/vllm.py:261`  `cfg.kernel_config.ir_op_priority.fused_add_rms_norm[0] == "aiter"`
- `vllm/kernels/oink_ops.py:105`  `Oink fused_add_rms_norm has the same constraints as rms_norm,`
- `vllm/kernels/oink_ops.py:110`  `@ir.ops.fused_add_rms_norm.register_impl(`
- `vllm/kernels/oink_ops.py:113`  `supported=has_oink_op("fused_add_rms_norm"),`
- `vllm/kernels/oink_ops.py:116`  `def fused_add_rms_norm(`
- `vllm/kernels/oink_ops.py:126`  `torch.ops.oink.fused_add_rms_norm(x_2d, residual_2d, weight, epsilon)`
- `vllm/kernels/vllm_c.py:54`  `@ir.ops.fused_add_rms_norm.register_impl(`
- `vllm/kernels/vllm_c.py:60`  `def fused_add_rms_norm(`
- `vllm/kernels/vllm_c.py:69`  `output, residual = ir.ops.fused_add_rms_norm.impls["native"].impl_fn(`
- `vllm/kernels/vllm_c.py:83`  `torch.ops._C.fused_add_rms_norm(x, x_residual, weight, epsilon)`
- `vllm/kernels/vllm_c.py:86`  `torch.ops._C.fused_add_rms_norm(x, x_residual, weight, epsilon)`
- `vllm/kernels/aiter_ops.py:111`  `AITER fused_add_rms_norm only supports 16-bit activations and no var_size override.`
- `vllm/kernels/aiter_ops.py:117`  `@ir.ops.fused_add_rms_norm.register_impl(`
- `vllm/kernels/aiter_ops.py:120`  `def fused_add_rms_norm(`
- `vllm/kernels/aiter_ops.py:131`  `return torch.ops.vllm_aiter.fused_add_rms_norm(x, x_residual, weight, epsilon)`
- `vllm/kernels/aiter_ops.py:179`  `op_name="fused_add_rms_norm",`
- `vllm/model_executor/custom_op.py:144`  `def forward_cuda(self, *args, **kwargs):`
- `vllm/platforms/cuda.py:773`  `fused_add_rms_norm=rms_norm,`
- `vllm/platforms/rocm.py:1471`  `fused_add_rms_norm=rms_norm,`
- `vllm/compilation/passes/pass_manager.py:172`  `# AR+RMS, since both consume fused_add_rms_norm.`
- `vllm/compilation/passes/fusion/add_rms_fusion.py:32`  `rms = vllm.ir.ops.rms_norm(residual_out, weight, self.epsilon)`
- `vllm/compilation/passes/fusion/add_rms_fusion.py:44`  `return vllm.ir.ops.fused_add_rms_norm(`
- `vllm/compilation/passes/fusion/add_rms_fusion.py:70`  `rms = vllm.ir.ops.rms_norm(input, weight, self.epsilon)`
- `vllm/compilation/passes/fusion/add_rms_fusion.py:82`  `return vllm.ir.ops.rms_norm(input, weight, self.epsilon)`
- `vllm/compilation/passes/fusion/add_rms_fusion.py:106`  `rms, residual_out = vllm.ir.ops.fused_add_rms_norm(`
- `vllm/compilation/passes/fusion/add_rms_fusion.py:122`  `rms, residual_out = vllm.ir.ops.fused_add_rms_norm(`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:44`  `_FUSED_ADD_RMS_NORM_OP = torch.ops.vllm_ir.fused_add_rms_norm.default`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:48`  `def _flashinfer_fused_add_rms_norm_nvfp4_quant(`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:77`  `def _flashinfer_fused_add_rms_norm_nvfp4_quant_fake(`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:104`  `op_name="flashinfer_fused_add_rms_norm_nvfp4_quant",`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:105`  `op_func=_flashinfer_fused_add_rms_norm_nvfp4_quant,`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:107`  `fake_impl=_flashinfer_fused_add_rms_norm_nvfp4_quant_fake,`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:110`  `torch.ops.vllm.flashinfer_fused_add_rms_norm_nvfp4_quant.default`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:123`  `# fused_add_rms_norm(x, residual, weight, epsilon, variance_size)`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:156`  `RMS_ADD_OP = torch.ops._C.fused_add_rms_norm.default`
- ...

## INT8 activation quant op (W8A8)
pattern: `scaled_int8_quant` (17 hits)
- `vllm/_cpu_ops.py:852`  `def onednn_scaled_int8_quant(`
- `vllm/_cpu_ops.py:881`  `torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)`
- `vllm/_cpu_ops.py:887`  `torch.ops._C.dynamic_scaled_int8_quant(output, input, input_scales, input_azp)`
- `vllm/_custom_ops.py:1873`  `def scaled_int8_quant(`
- `vllm/_custom_ops.py:1916`  `torch.ops._C.static_scaled_int8_quant(output, input, scale, azp)`
- `vllm/_custom_ops.py:1924`  `torch.ops._C.dynamic_scaled_int8_quant(`
- `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:125`  `# ops.scaled_int8_quant supports both dynamic and static quant:`
- `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:129`  `x_q, x_s, x_zp = ops.scaled_int8_quant(`
- `vllm/model_executor/kernels/linear/scaled_mm/triton.py:127`  `x_q, x_s, x_zp = ops.scaled_int8_quant(`
- `vllm/model_executor/kernels/linear/scaled_mm/aiter.py:96`  `# ops.scaled_int8_quant supports both dynamic and static quant:`
- `vllm/model_executor/kernels/linear/scaled_mm/aiter.py:103`  `x_q, x_s, x_zp = ops.scaled_int8_quant(x, i_s, i_zp, symmetric=symmetric)`
- `vllm/model_executor/kernels/linear/scaled_mm/cpu.py:198`  `# ops.scaled_int8_quant supports both dynamic and static quant:`
- `vllm/model_executor/kernels/linear/scaled_mm/cpu.py:201`  `x_q, x_s, x_zp = ops.onednn_scaled_int8_quant(`
- `vllm/model_executor/layers/fused_moe/utils.py:365`  `A, A_scale, _ = ops.scaled_int8_quant(A, scale=A_scale)`
- `vllm/model_executor/layers/fused_moe/utils.py:369`  `A, A_scale, _ = ops.scaled_int8_quant(A, scale=A_scale)`
- `vllm/model_executor/layers/fused_moe/experts/triton_moe.py:176`  `# `scaled_int8_quant`, which is plain elementwise arithmetic.`
- `vllm/model_executor/layers/quantization/quark/quark_moe.py:1297`  `ops.scaled_int8_quant(`

## Existing fused CUDA kernel binding
pattern: `rms_norm_dynamic_per_token_quant` (6 hits)
- `vllm/_custom_ops.py:397`  `def rms_norm_dynamic_per_token_quant(`
- `vllm/_custom_ops.py:410`  `torch.ops._C.rms_norm_dynamic_per_token_quant(`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:195`  `): torch.ops._C.rms_norm_dynamic_per_token_quant.default,  # noqa: E501`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:198`  `): torch.ops._C.rms_norm_dynamic_per_token_quant.default,  # noqa: E501`
- `vllm/compilation/passes/utility/fix_functionalization.py:104`  `elif at_target == torch.ops._C.rms_norm_dynamic_per_token_quant.default:  # noqa: E501`
- `vllm/kernels/helion/ops/rms_norm_dynamic_per_token_quant.py:174`  `def rms_norm_dynamic_per_token_quant(`

## W8A8 INT8 linear scheme (compressed-tensors)
pattern: `CompressedTensorsW8A8Int8` (10 hits)
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:49`  `CompressedTensorsW8A8Int8,`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:923`  `return CompressedTensorsW8A8Int8(`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors.py:930`  `return CompressedTensorsW8A8Int8(`
- `vllm/model_executor/layers/quantization/quark/quark_moe.py:1070`  `# activation scale (this matches CompressedTensorsW8A8Int8MoEMethod).`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py:175`  `CompressedTensorsW8A8Int8MoEMethod,`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe.py:178`  `return CompressedTensorsW8A8Int8MoEMethod(`
- `vllm/model_executor/layers/quantization/compressed_tensors/compressed_tensors_moe/compressed_tensors_moe_w8a8_int8.py:38`  `class CompressedTensorsW8A8Int8MoEMethod(CompressedTensorsMoEMethod):`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/__init__.py:11`  `from .compressed_tensors_w8a8_int8 import CompressedTensorsW8A8Int8`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/__init__.py:24`  `"CompressedTensorsW8A8Int8",`
- `vllm/model_executor/layers/quantization/compressed_tensors/schemes/compressed_tensors_w8a8_int8.py:26`  `class CompressedTensorsW8A8Int8(CompressedTensorsScheme):`

## INT8 scaled-mm kernels (where quant is called)
pattern: `CutlassScaledMMLinearKernel|Int8ScaledMM|class .*ScaledMM` (111 hits)
- `vllm/_aiter_ops.py:892`  `# CutlassInt8ScaledMMLinearKernel prepare weight `w_q` in [K, N] format`
- `vllm/model_executor/kernels/linear/__init__.py:180`  `Int8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:181`  `Int8ScaledMMLinearLayerConfig,`
- `vllm/model_executor/kernels/linear/__init__.py:187`  `AiterInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:201`  `CPUInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:206`  `CutlassInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:217`  `HummingInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:234`  `TritonInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:242`  `ZentorchInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:297`  `CutlassInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:327`  `HummingInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:342`  `TritonInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:360`  `AiterInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:447`  `_POSSIBLE_INT8_KERNELS: dict[PlatformEnum, list[type[Int8ScaledMMLinearKernel]]] = {`
- `vllm/model_executor/kernels/linear/__init__.py:448`  `PlatformEnum.CPU: [ZentorchInt8ScaledMMLinearKernel, CPUInt8ScaledMMLinearKernel],`
- `vllm/model_executor/kernels/linear/__init__.py:450`  `CutlassInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:451`  `TritonInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:452`  `HummingInt8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/__init__.py:454`  `PlatformEnum.ROCM: [AiterInt8ScaledMMLinearKernel, TritonInt8ScaledMMLinearKernel],`
- `vllm/model_executor/kernels/linear/__init__.py:455`  `PlatformEnum.XPU: [TritonInt8ScaledMMLinearKernel],`
- `vllm/model_executor/kernels/linear/__init__.py:858`  `) -> Int8ScaledMMLinearKernel:`
- `vllm/model_executor/kernels/linear/__init__.py:859`  `config = Int8ScaledMMLinearLayerConfig(`
- `vllm/model_executor/kernels/linear/__init__.py:1377`  `"Int8ScaledMMLinearKernel",`
- `vllm/model_executor/kernels/linear/__init__.py:1380`  `"Int8ScaledMMLinearLayerConfig",`
- `vllm/model_executor/kernels/linear/__init__.py:1387`  `"AiterInt8ScaledMMLinearKernel",`
- `vllm/model_executor/kernels/linear/__init__.py:1388`  `"CPUInt8ScaledMMLinearKernel",`
- `vllm/model_executor/kernels/linear/__init__.py:1390`  `"CutlassInt8ScaledMMLinearKernel",`
- `vllm/model_executor/kernels/linear/__init__.py:1396`  `"TritonInt8ScaledMMLinearKernel",`
- `vllm/model_executor/kernels/linear/__init__.py:1397`  `"ZentorchInt8ScaledMMLinearKernel",`
- `vllm/model_executor/kernels/linear/scaled_mm/humming.py:20`  `Int8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/scaled_mm/humming.py:21`  `Int8ScaledMMLinearLayerConfig,`
- `vllm/model_executor/kernels/linear/scaled_mm/humming.py:27`  `class HummingFP8ScaledMMLinearKernel(FP8ScaledMMLinearKernel):`
- `vllm/model_executor/kernels/linear/scaled_mm/humming.py:118`  `class HummingInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):`
- `vllm/model_executor/kernels/linear/scaled_mm/humming.py:138`  `cls, config: Int8ScaledMMLinearLayerConfig`
- `vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:57`  `class Fp8BlockScaledMMLinearKernel(`
- `vllm/model_executor/kernels/linear/scaled_mm/BlockScaledMMLinearKernel.py:153`  `class Fp8BlockScaledDynamicMMLinearKernel(Fp8BlockScaledMMLinearKernel, ABC):`
- `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:28`  `Int8ScaledMMLinearKernel,`
- `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:29`  `Int8ScaledMMLinearLayerConfig,`
- `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:33`  `class CutlassInt8ScaledMMLinearKernel(Int8ScaledMMLinearKernel):`
- `vllm/model_executor/kernels/linear/scaled_mm/cutlass.py:43`  `def can_implement(cls, c: Int8ScaledMMLinearLayerConfig) -> tuple[bool, str | None]:`
- ...

## Qwen3 decoder layer (norm -> linear call sites)
pattern: `class Qwen3DecoderLayer|input_layernorm|post_attention_layernorm` (636 hits)
- `vllm/model_executor/models/glm4v.py:195`  `self.input_layernorm = LayerNorm(config.hidden_size, eps=config.layer_norm_eps)`
- `vllm/model_executor/models/glm4v.py:202`  `self.post_attention_layernorm = LayerNorm(`
- `vllm/model_executor/models/glm4v.py:208`  `attention_output = self.input_layernorm(self.attention(attention_input))`
- `vllm/model_executor/models/glm4v.py:211`  `mlp_output = self.post_attention_layernorm(self.mlp(mlp_input))`
- `vllm/model_executor/models/mimo_v2.py:477`  `self.input_layernorm = RMSNorm(config.hidden_size, eps=config.layernorm_epsilon)`
- `vllm/model_executor/models/mimo_v2.py:478`  `self.post_attention_layernorm = RMSNorm(`
- `vllm/model_executor/models/mimo_v2.py:490`  `hidden_states = self.input_layernorm(hidden_states)`
- `vllm/model_executor/models/mimo_v2.py:492`  `hidden_states, residual = self.input_layernorm(hidden_states, residual)`
- `vllm/model_executor/models/mimo_v2.py:499`  `hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)`
- `vllm/model_executor/models/llama_eagle.py:47`  `disable_input_layernorm: bool,`
- `vllm/model_executor/models/llama_eagle.py:53`  `# Skip the input_layernorm`
- `vllm/model_executor/models/llama_eagle.py:55`  `if disable_input_layernorm:`
- `vllm/model_executor/models/llama_eagle.py:56`  `del self.input_layernorm`
- `vllm/model_executor/models/llama_eagle.py:57`  `self.input_layernorm = nn.Identity()`
- `vllm/model_executor/models/sarvam.py:397`  `self.input_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)`
- `vllm/model_executor/models/sarvam.py:405`  `self.post_attention_layernorm = RMSNorm(hidden_size, eps=config.rms_norm_eps)`
- `vllm/model_executor/models/sarvam.py:440`  `hidden_states = self.input_layernorm(hidden_states)`
- `vllm/model_executor/models/sarvam.py:442`  `hidden_states, residual = self.input_layernorm(hidden_states, residual)`
- `vllm/model_executor/models/sarvam.py:448`  `hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)`
- `vllm/model_executor/models/hy_v3.py:389`  `self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)`
- `vllm/model_executor/models/hy_v3.py:390`  `self.post_attention_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)`
- `vllm/model_executor/models/hy_v3.py:417`  `hidden_states = self.input_layernorm(hidden_states)`
- `vllm/model_executor/models/hy_v3.py:419`  `hidden_states, residual = self.input_layernorm(hidden_states, residual)`
- `vllm/model_executor/models/hy_v3.py:426`  `hidden_states, residual = self.post_attention_layernorm(hidden_states, residual)`
- `vllm/model_executor/models/granitemoehybrid.py:109`  `self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)`
- `vllm/model_executor/models/granitemoehybrid.py:110`  `self.post_attention_layernorm = RMSNorm(`
- `vllm/model_executor/models/granitemoehybrid.py:121`  `hidden_states = self.input_layernorm(hidden_states)`
- `vllm/model_executor/models/granitemoehybrid.py:126`  `hidden_states = self.post_attention_layernorm(hidden_states)`
- `vllm/model_executor/models/granitemoehybrid.py:185`  `self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)`
- `vllm/model_executor/models/granitemoehybrid.py:186`  `self.post_attention_layernorm = RMSNorm(`
- `vllm/model_executor/models/granitemoehybrid.py:197`  `hidden_states = self.input_layernorm(hidden_states)`
- `vllm/model_executor/models/granitemoehybrid.py:206`  `hidden_states = self.post_attention_layernorm(hidden_states)`
- `vllm/model_executor/models/longcat_flash.py:291`  `self.input_layernorm = nn.ModuleList(`
- `vllm/model_executor/models/longcat_flash.py:294`  `self.post_attention_layernorm = nn.ModuleList(`
- `vllm/model_executor/models/longcat_flash.py:332`  `hidden_states = self.input_layernorm[0](hidden_states)`
- `vllm/model_executor/models/longcat_flash.py:334`  `hidden_states, residual = self.input_layernorm[0](hidden_states, residual)`
- `vllm/model_executor/models/longcat_flash.py:342`  `hidden_states, residual = self.post_attention_layernorm[0](`
- `vllm/model_executor/models/longcat_flash.py:353`  `hidden_states, residual = self.input_layernorm[1](hidden_states, residual)`
- `vllm/model_executor/models/longcat_flash.py:361`  `hidden_states, residual = self.post_attention_layernorm[1](`
- `vllm/model_executor/models/bailing_moe_v3.py:1282`  `self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)`
- ...

## Fusion passes (torch.compile pattern matcher)
pattern: `RMSNormQuantFusionPass|class .*FusionPass|QuantKey|kInt8|rms_quant` (887 hits)
- `vllm/config/quantization.py:23`  `QuantKey,`
- `vllm/config/quantization.py:31`  `kInt8Dynamic32Sym,`
- `vllm/config/quantization.py:32`  `kInt8StaticChannelSym,`
- `vllm/config/quantization.py:43`  `QUANT_KEY_NAMES: dict[str, QuantKey] = {`
- `vllm/config/quantization.py:53`  `"int8_per_channel_static": kInt8StaticChannelSym,`
- `vllm/config/quantization.py:56`  `"int8_per_group_32": kInt8Dynamic32Sym,`
- `vllm/config/quantization.py:63`  `_WEIGHT_QUANT_KEY_NAMES: dict[str, QuantKey] = {`
- `vllm/config/quantization.py:69`  `def _coerce_quant_key(v: Any) -> QuantKey | None:`
- `vllm/config/quantization.py:70`  `if v is None or isinstance(v, QuantKey):`
- `vllm/config/quantization.py:73`  `raise TypeError(f"expected str or QuantKey, got {type(v).__name__}")`
- `vllm/config/quantization.py:83`  `# Stop pydantic from introspecting QuantKey: it transitively contains a`
- `vllm/config/quantization.py:85`  `QuantKeyField = Annotated[`
- `vllm/config/quantization.py:86`  `QuantKey | None,`
- `vllm/config/quantization.py:94`  `_UNSET = cast(QuantKey | None, object())`
- `vllm/config/quantization.py:97`  `def quant_key_name(quant_key: QuantKey | None) -> str | None:`
- `vllm/config/quantization.py:99`  `# TODO: QuantKey itself should define `__str__`, instead of having this logic.`
- `vllm/config/quantization.py:116`  `weight: QuantKeyField = _UNSET`
- `vllm/config/quantization.py:119`  `activation: QuantKeyField = _UNSET`
- `vllm/config/quantization.py:302`  `# defaults to activation_quant_key=kInt8DynamicTokenSym.`
- `vllm/config/quantization.py:304`  `moe=QuantSpec(weight=kInt8StaticChannelSym),`
- `vllm/config/quantization.py:430`  `def user_activation_override(kind: str, config: Any) -> tuple[QuantKey | None, bool]:`
- `vllm/compilation/passes/pass_manager.py:27`  `RocmAiterRMSNormQuantFusionPass,`
- `vllm/compilation/passes/pass_manager.py:46`  `from .fusion.rms_quant_fusion import RMSNormQuantFusionPass`
- `vllm/compilation/passes/pass_manager.py:63`  `from .fusion.rms_quant_fusion import RMSNormQuantFusionPass`
- `vllm/compilation/passes/pass_manager.py:192`  `RocmAiterRMSNormQuantFusionPass(config),`
- `vllm/compilation/passes/pass_manager.py:194`  `self.passes += [RMSNormQuantFusionPass(config)]`
- `vllm/compilation/passes/fusion/attn_quant_fusion.py:15`  `QuantKey,`
- `vllm/compilation/passes/fusion/attn_quant_fusion.py:25`  `from .rms_quant_fusion import QUANT_OPS`
- `vllm/compilation/passes/fusion/attn_quant_fusion.py:36`  `_FP8_QUANT_KEY = QuantKey(dtype=FP8_DTYPE, scale=kStaticTensorScale, symmetric=True)`
- `vllm/compilation/passes/fusion/attn_quant_fusion.py:405`  `class AttnQuantFusionPass(VllmFusionPatternMatcherPass):`
- `vllm/compilation/passes/fusion/add_rms_fusion.py:140`  `class AddRMSNormFusionPass(VllmFusionPatternMatcherPass):`
- `vllm/compilation/passes/fusion/add_rms_fusion.py:152`  `class RMSNormReshapeFusionPass(VllmFusionPatternMatcherPass):`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:17`  `QuantKey,`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:158`  `QUANT_OPS: dict[QuantKey, OpOverload] = {`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:170`  `class FusedRMSQuantKey(NamedTuple):`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:176`  `quant: QuantKey`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:181`  `f"FusedQuantKey({self.quant}, with"`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:186`  `FUSED_OPS: dict[FusedRMSQuantKey, OpOverload] = {`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:187`  `FusedRMSQuantKey(`
- `vllm/compilation/passes/fusion/rms_quant_fusion.py:190`  `FusedRMSQuantKey(`
- ...

## Custom-op registration (for a new Triton op)
pattern: `direct_register_custom_op` (282 hits)
- `vllm/_aiter_ops.py:17`  `from vllm.utils.torch_utils import direct_register_custom_op`
- `vllm/_aiter_ops.py:3060`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3072`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3080`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3088`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3095`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3103`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3110`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3117`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3125`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3132`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3138`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3145`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3151`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3157`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3163`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3169`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3175`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3181`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3187`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3194`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3201`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3207`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3213`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3219`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3225`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3231`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3237`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3244`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3250`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3257`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3264`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3272`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3280`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3289`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3295`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3301`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3307`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3313`  `direct_register_custom_op(`
- `vllm/_aiter_ops.py:3320`  `direct_register_custom_op(`
- ...

## Out-of-tree plugin entry points
pattern: `vllm\.general_plugins|load_general_plugins|register_oot` (42 hits)
- `vllm/device_allocator/sleep_mode_backend.py:17`  `backends register through a ``vllm.general_plugins`` entry point at import time.`
- `vllm/device_allocator/sleep_mode_backend.py:160`  `the same way from a ``vllm.general_plugins`` entry point.`
- `vllm/device_allocator/sleep_mode_backend.py:203`  `# durable snapshot) register the same way through a vllm.general_plugins entry`
- `vllm/plugins/__init__.py:18`  `DEFAULT_PLUGINS_GROUP = "vllm.general_plugins"`
- `vllm/plugins/__init__.py:77`  `def load_general_plugins():`
- `vllm/engine/arg_utils.py:123`  `from vllm.plugins import load_general_plugins`
- `vllm/engine/arg_utils.py:936`  `from vllm.plugins import load_general_plugins`
- `vllm/engine/arg_utils.py:938`  `load_general_plugins()`
- `vllm/engine/arg_utils.py:3293`  `load_general_plugins()`
- `vllm/model_executor/custom_op.py:83`  `def register_oot(cls, _decorated_layer_cls=None, name: str | None = None):`
- `vllm/model_executor/custom_op.py:92`  `# Called with parentheses: @PluggableLayer.register_oot()`
- `vllm/model_executor/custom_op.py:93`  `# or @PluggableLayer.register_oot(name="...")`
- `vllm/model_executor/custom_op.py:96`  `# Called without parentheses: @PluggableLayer.register_oot`
- `vllm/model_executor/custom_op.py:330`  `# - @UnquantizedFusedMoEMethod.register_oot`
- `vllm/model_executor/custom_op.py:333`  `# - @CustomOP.register_oot(name="UnquantizedFusedMoEMethod")`
- `vllm/model_executor/custom_op.py:335`  `def register_oot(cls, _decorated_op_cls=None, name: str | None = None):`
- `vllm/model_executor/custom_op.py:344`  `# Called with parentheses: @CustomOP.register_oot()`
- `vllm/model_executor/custom_op.py:345`  `# or @CustomOP.register_oot(name="...")`
- `vllm/model_executor/custom_op.py:350`  `# Called without parentheses: @CustomOP.register_oot`
- `vllm/model_executor/models/registry.py:1726`  `from vllm.plugins import load_general_plugins`
- `vllm/model_executor/models/registry.py:1728`  `load_general_plugins()`
- `vllm/model_executor/layers/mla.py:47`  `Note that currently oot platforms can still use CustomOp.register_oot to`
- `vllm/model_executor/hw_agnostic/custom_op.py:54`  `"class's register_oot.",`
- `vllm/model_executor/hw_agnostic/custom_op.py:113`  `def register_oot(cls, _decorated_layer_cls=None, name: str | None = None):`
- `vllm/model_executor/hw_agnostic/custom_op.py:122`  `# Called with parentheses: @PluggableLayer.register_oot()`
- `vllm/model_executor/hw_agnostic/custom_op.py:123`  `# or @PluggableLayer.register_oot(name="...")`
- `vllm/model_executor/hw_agnostic/custom_op.py:126`  `# Called without parentheses: @PluggableLayer.register_oot`
- `vllm/model_executor/hw_agnostic/custom_op.py:169`  `# an OOT plugin via register_oot).`
- `vllm/model_executor/hw_agnostic/custom_op.py:309`  `# - @UnquantizedFusedMoEMethod.register_oot`
- `vllm/model_executor/hw_agnostic/custom_op.py:312`  `# - @CustomOP.register_oot(name="UnquantizedFusedMoEMethod")`
- `vllm/model_executor/hw_agnostic/custom_op.py:314`  `def register_oot(cls, _decorated_op_cls=None, name: str | None = None):`
- `vllm/model_executor/hw_agnostic/custom_op.py:323`  `# Called with parentheses: @CustomOP.register_oot()`
- `vllm/model_executor/hw_agnostic/custom_op.py:324`  `# or @CustomOP.register_oot(name="...")`
- `vllm/model_executor/hw_agnostic/custom_op.py:329`  `# Called without parentheses: @CustomOP.register_oot`
- `vllm/model_executor/hw_agnostic/_resolve.py:41`  ``register_oot` overrides the class that backend builds on either path.`
- `vllm/model_executor/hw_agnostic/_resolve.py:49`  `@SiluAndMul.register_oot`
- `vllm/model_executor/layers/rotary_embedding/mrope_vit_setup.py:104`  `platforms replace the whole layer via ``@VitMropeSetup.register_oot```
- `vllm/plugins/endpoint_plugins/interface.py:13`  ``vllm.general_plugins` (see `vllm/plugins/__init__.py`). The`
- `vllm/v1/worker/worker_base.py:303`  `from vllm.plugins import load_general_plugins`
- `vllm/v1/worker/worker_base.py:305`  `load_general_plugins()`
- ...
