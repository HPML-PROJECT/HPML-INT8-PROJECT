# Part 1 vs Part 2 alignment (BF16, residual add)

Part 1 = CUDA-graph median (µs, locked clocks). Part 2 = Nsight Compute kernel time (µs, sum of V0's two kernels).
Ratio = V1 / V0 (>1: fused is slower). 'agree' = both ratios on the same side of 1 (±10%).

| GPU | H | T | cache | P1 V0 | P1 V1 | P1 ratio | ncu V0 | ncu V1 | ncu ratio | agree |
|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA_L4 | 2048 | 8 | cold | 7.2 | 7.2 | 1.00 | 6.3 | 4.3 | 0.68 | **NO** |
| NVIDIA_L4 | 2048 | 8 | warm | 3.4 | 2.7 | 0.79 | 5.2 | 3.6 | 0.70 | yes |
| NVIDIA_L4 | 2048 | 256 | cold | 23.6 | 22.5 | 0.96 | 18.1 | 15.1 | 0.83 | **NO** |
| NVIDIA_L4 | 2048 | 256 | warm | 5.4 | 10.7 | 1.97 | 7.2 | 11.8 | 1.65 | yes |
| NVIDIA_L4 | 2048 | 1024 | cold | 86.0 | 74.8 | 0.87 | 53.4 | 45.4 | 0.85 | yes |
| NVIDIA_L4 | 2048 | 1024 | warm | 16.6 | 38.5 | 2.31 | 16.3 | 38.3 | 2.35 | yes |
| NVIDIA_L4 | 2048 | 4096 | cold | 326.7 | 263.2 | 0.81 | 217.0 | 171.8 | 0.79 | yes |
| NVIDIA_L4 | 2048 | 4096 | warm | 109.2 | 173.2 | 1.59 | 104.7 | 168.1 | 1.61 | yes |
| NVIDIA_L4 | 2048 | 16384 | cold | 1588.2 | 969.7 | 0.61 | 1303.2 | 833.9 | 0.64 | yes |
| NVIDIA_L4 | 2048 | 16384 | warm | 1523.7 | 999.7 | 0.66 | 1532.3 | 897.3 | 0.59 | yes |
| NVIDIA_L4 | 4096 | 8 | cold | 9.2 | 8.2 | 0.89 | 6.7 | 5.0 | 0.74 | yes |
| NVIDIA_L4 | 4096 | 8 | warm | 3.8 | 3.2 | 0.84 | 5.6 | 4.2 | 0.75 | yes |
| NVIDIA_L4 | 4096 | 256 | cold | 42.0 | 37.9 | 0.90 | 30.9 | 21.9 | 0.71 | **NO** |
| NVIDIA_L4 | 4096 | 256 | warm | 8.4 | 14.7 | 1.75 | 10.9 | 15.1 | 1.39 | yes |
| NVIDIA_L4 | 4096 | 1024 | cold | 166.9 | 140.3 | 0.84 | 103.3 | 68.1 | 0.66 | yes |
| NVIDIA_L4 | 4096 | 1024 | warm | 36.7 | 53.5 | 1.46 | 26.5 | 49.6 | 1.87 | yes |
| NVIDIA_L4 | 4096 | 4096 | cold | 741.4 | 516.1 | 0.70 | 556.3 | 346.9 | 0.62 | yes |
| NVIDIA_L4 | 4096 | 4096 | warm | 716.3 | 491.8 | 0.69 | 628.4 | 471.1 | 0.75 | yes |
| NVIDIA_L4 | 4096 | 16384 | cold | 3130.4 | 2046.0 | 0.65 | 2869.2 | 1871.0 | 0.65 | yes |
| NVIDIA_L4 | 4096 | 16384 | warm | 3076.1 | 2006.0 | 0.65 | 3104.7 | 2011.5 | 0.65 | yes |

17/20 configurations agree on direction.
Disagreements usually mean the profiler's replay changed the cache state (most likely in 'warm' mode, where V0's second kernel may lose the L2 reuse it gets in a real back-to-back run). Check L2 hit rates.
