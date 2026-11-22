# Initial bottleneck analysis (auto-generated; verify against the raw tables)

Ratios are V1 (fused CUDA) / V0 (unfused pair, summed). >1 means the fused kernel is worse.

| GPU | H | cache | T | V0 us | V1 us | time ratio | DRAM MB ratio | V1 DRAM %peak | V1 L2 hit % | V1 regs | V1 occ % | V0 occ % | V1 top stall |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| NVIDIA_L4 | 2048 | cold | 8 | 6.3 | 4.26 | 0.68 | 0.81 | 8.8 | 67.9 | 32.0 | 63.9 | 57.8 | barrier |
| NVIDIA_L4 | 2048 | cold | 256 | 18.14 | 15.1 | 0.83 | 0.66 | 51.7 | 48.3 | 32.0 | 62.8 | 69.3 | barrier |
| NVIDIA_L4 | 2048 | cold | 1024 | 53.41 | 45.44 | 0.85 | 0.66 | 67.9 | 44.6 | 32.0 | 62.0 | 90.4 | barrier |
| NVIDIA_L4 | 2048 | cold | 4096 | 216.96 | 171.78 | 0.79 | 0.65 | 78.4 | 43.4 | 32.0 | 60.7 | 90.5 | barrier |
| NVIDIA_L4 | 2048 | cold | 16384 | 1303.17 | 833.89 | 0.64 | 0.64 | 92.6 | 43.1 | 32.0 | 62.9 | 93.7 | barrier |
| NVIDIA_L4 | 2048 | warm | 8 | 5.18 | 3.62 | 0.70 | nan | 1.4 | 50.9 | 32.0 | 55.9 | 48.5 | barrier |
| NVIDIA_L4 | 2048 | warm | 256 | 7.17 | 11.84 | 1.65 | 0.38 | 2.5 | 96.1 | 32.0 | 60.6 | 61.5 | barrier |
| NVIDIA_L4 | 2048 | warm | 1024 | 16.32 | 38.34 | 2.35 | 0.38 | 0.2 | 98.9 | 32.0 | 61.2 | 86.7 | barrier |
| NVIDIA_L4 | 2048 | warm | 4096 | 104.7 | 168.1 | 1.61 | 1.34 | 63.9 | 71.5 | 32.0 | 60.4 | 94.9 | barrier |
| NVIDIA_L4 | 2048 | warm | 16384 | 1532.29 | 897.34 | 0.59 | 0.62 | 95.5 | 43.0 | 32.0 | 64.6 | 96.4 | barrier |
| NVIDIA_L4 | 4096 | cold | 8 | 6.69 | 4.96 | 0.74 | 0.75 | 12.6 | 65.4 | 32.0 | 65.8 | 57.2 | long_scoreboard |
| NVIDIA_L4 | 4096 | cold | 256 | 30.91 | 21.86 | 0.71 | 0.65 | 70.7 | 47.3 | 32.0 | 65.4 | 71.3 | long_scoreboard |
| NVIDIA_L4 | 4096 | cold | 1024 | 103.3 | 68.1 | 0.66 | 0.65 | 89.2 | 44.5 | 32.0 | 65.2 | 90.8 | long_scoreboard |
| NVIDIA_L4 | 4096 | cold | 4096 | 556.29 | 346.88 | 0.62 | 0.62 | 92.7 | 43.5 | 32.0 | 63.9 | 89.9 | long_scoreboard |
| NVIDIA_L4 | 4096 | cold | 16384 | 2869.15 | 1871.04 | 0.65 | 0.62 | 87.7 | 43.0 | 32.0 | 65.1 | 91.8 | barrier |
| NVIDIA_L4 | 4096 | warm | 8 | 5.57 | 4.16 | 0.75 | nan | 0.1 | 66.2 | 32.0 | 59.2 | 51.6 | barrier |
| NVIDIA_L4 | 4096 | warm | 256 | 10.88 | 15.1 | 1.39 | 0.82 | 0.3 | 98.1 | 32.0 | 62.2 | 64.5 | barrier |
| NVIDIA_L4 | 4096 | warm | 1024 | 26.53 | 49.63 | 1.87 | 0.18 | 0.3 | 99.4 | 32.0 | 62.9 | 86.5 | barrier |
| NVIDIA_L4 | 4096 | warm | 4096 | 628.38 | 471.07 | 0.75 | 0.73 | 90.0 | 43.5 | 32.0 | 66.5 | 93.6 | barrier |
| NVIDIA_L4 | 4096 | warm | 16384 | 3104.67 | 2011.46 | 0.65 | 0.62 | 85.5 | 43.0 | 32.0 | 66.3 | 93.0 | barrier |

## Observations per configuration
- **NVIDIA_L4, H=2048, T=8, cold**: time ratio 0.68 (fused FASTER); V1 uses only 8.8% of peak DRAM throughput -> not bandwidth-saturated; V1 executes 2.5x the instructions of V0's heaviest kernel
- **NVIDIA_L4, H=2048, T=256, cold**: time ratio 0.83 (fused FASTER); V1 uses only 51.7% of peak DRAM throughput -> not bandwidth-saturated; V1 executes 4.2x the instructions of V0's heaviest kernel
- **NVIDIA_L4, H=2048, T=1024, cold**: time ratio 0.85 (fused FASTER); V1 executes 4.2x the instructions of V0's heaviest kernel; V1 achieved occupancy 62.0% < V0 90.4% (regs/thread 32.0)
- **NVIDIA_L4, H=2048, T=4096, cold**: time ratio 0.79 (fused FASTER); V1 executes 4.2x the instructions of V0's heaviest kernel; V1 achieved occupancy 60.7% < V0 90.5% (regs/thread 32.0)
- **NVIDIA_L4, H=2048, T=16384, cold**: time ratio 0.64 (fused FASTER); V1 executes 4.2x the instructions of V0's heaviest kernel; V1 achieved occupancy 62.9% < V0 93.7% (regs/thread 32.0)
- **NVIDIA_L4, H=2048, T=8, warm**: time ratio 0.70 (fused FASTER); V1 uses only 1.4% of peak DRAM throughput -> not bandwidth-saturated; V1 executes 2.5x the instructions of V0's heaviest kernel
- **NVIDIA_L4, H=2048, T=256, warm**: time ratio 1.65 (fused SLOWER); V1 uses only 2.5% of peak DRAM throughput -> not bandwidth-saturated; V1 executes 4.2x the instructions of V0's heaviest kernel
- **NVIDIA_L4, H=2048, T=1024, warm**: time ratio 2.35 (fused SLOWER); V1 uses only 0.2% of peak DRAM throughput -> not bandwidth-saturated; V1 executes 4.2x the instructions of V0's heaviest kernel; V1 achieved occupancy 61.2% < V0 86.7% (regs/thread 32.0)
- **NVIDIA_L4, H=2048, T=4096, warm**: time ratio 1.61 (fused SLOWER); V1 executes 4.2x the instructions of V0's heaviest kernel; V1 achieved occupancy 60.4% < V0 94.9% (regs/thread 32.0); V1 moves 1.34x the DRAM bytes of V0
- **NVIDIA_L4, H=2048, T=16384, warm**: time ratio 0.59 (fused FASTER); V1 executes 4.2x the instructions of V0's heaviest kernel; V1 achieved occupancy 64.6% < V0 96.4% (regs/thread 32.0)
- **NVIDIA_L4, H=4096, T=8, cold**: time ratio 0.74 (fused FASTER); V1 uses only 12.6% of peak DRAM throughput -> not bandwidth-saturated; top V1 stall is long_scoreboard (memory-latency/LSU pressure); V1 executes 2.5x the instructions of V0's heaviest kernel
- **NVIDIA_L4, H=4096, T=256, cold**: time ratio 0.71 (fused FASTER); top V1 stall is long_scoreboard (memory-latency/LSU pressure); V1 executes 3.8x the instructions of V0's heaviest kernel
- **NVIDIA_L4, H=4096, T=1024, cold**: time ratio 0.66 (fused FASTER); top V1 stall is long_scoreboard (memory-latency/LSU pressure); V1 executes 3.8x the instructions of V0's heaviest kernel; V1 achieved occupancy 65.2% < V0 90.8% (regs/thread 32.0)
- **NVIDIA_L4, H=4096, T=4096, cold**: time ratio 0.62 (fused FASTER); top V1 stall is long_scoreboard (memory-latency/LSU pressure); V1 executes 3.8x the instructions of V0's heaviest kernel; V1 achieved occupancy 63.9% < V0 89.9% (regs/thread 32.0)
- **NVIDIA_L4, H=4096, T=16384, cold**: time ratio 0.65 (fused FASTER); V1 executes 3.8x the instructions of V0's heaviest kernel; V1 achieved occupancy 65.1% < V0 91.8% (regs/thread 32.0)
- **NVIDIA_L4, H=4096, T=8, warm**: time ratio 0.75 (fused FASTER); V1 uses only 0.1% of peak DRAM throughput -> not bandwidth-saturated; V1 executes 2.5x the instructions of V0's heaviest kernel
- **NVIDIA_L4, H=4096, T=256, warm**: time ratio 1.39 (fused SLOWER); V1 uses only 0.3% of peak DRAM throughput -> not bandwidth-saturated; V1 executes 3.8x the instructions of V0's heaviest kernel
- **NVIDIA_L4, H=4096, T=1024, warm**: time ratio 1.87 (fused SLOWER); V1 uses only 0.3% of peak DRAM throughput -> not bandwidth-saturated; V1 executes 3.8x the instructions of V0's heaviest kernel; V1 achieved occupancy 62.9% < V0 86.5% (regs/thread 32.0)
- **NVIDIA_L4, H=4096, T=4096, warm**: time ratio 0.75 (fused FASTER); V1 executes 3.8x the instructions of V0's heaviest kernel; V1 achieved occupancy 66.5% < V0 93.6% (regs/thread 32.0)
- **NVIDIA_L4, H=4096, T=16384, warm**: time ratio 0.65 (fused FASTER); V1 executes 3.8x the instructions of V0's heaviest kernel; V1 achieved occupancy 66.3% < V0 93.0% (regs/thread 32.0)

## H2: cache residency (warm/cold time per variant)
| GPU | H | T | V0 warm/cold | V1 warm/cold | V0 L2 hit warm % | V1 L2 hit warm % |
|---|---|---|---|---|---|---|
| NVIDIA_L4 | 2048 | 8 | 0.82 | 0.85 | std::enable_if<T2>0&&_ty: 79.3 / void dynamic_scaled_int8: 49.8 | void rms_norm_dynamic_pe: 50.9 |
| NVIDIA_L4 | 2048 | 256 | 0.40 | 0.78 | std::enable_if<T2>0&&_ty: 98.9 / void dynamic_scaled_int8: 96.1 | void rms_norm_dynamic_pe: 96.1 |
| NVIDIA_L4 | 2048 | 1024 | 0.31 | 0.84 | std::enable_if<T2>0&&_ty: 99.7 / void dynamic_scaled_int8: 99.3 | void rms_norm_dynamic_pe: 98.9 |
| NVIDIA_L4 | 2048 | 4096 | 0.48 | 0.98 | std::enable_if<T2>0&&_ty: 86.2 / void dynamic_scaled_int8: 99.9 | void rms_norm_dynamic_pe: 71.5 |
| NVIDIA_L4 | 2048 | 16384 | 1.18 | 1.08 | std::enable_if<T2>0&&_ty: 50.5 / void dynamic_scaled_int8: 35.9 | void rms_norm_dynamic_pe: 43.0 |
| NVIDIA_L4 | 4096 | 8 | 0.83 | 0.84 | std::enable_if<T2>0&&_ty: 88.1 / void dynamic_scaled_int8: 64.3 | void rms_norm_dynamic_pe: 66.2 |
| NVIDIA_L4 | 4096 | 256 | 0.35 | 0.69 | std::enable_if<T2>0&&_ty: 99.5 / void dynamic_scaled_int8: 98.0 | void rms_norm_dynamic_pe: 98.1 |
| NVIDIA_L4 | 4096 | 1024 | 0.26 | 0.73 | std::enable_if<T2>0&&_ty: 100.2 / void dynamic_scaled_int8: 99.6 | void rms_norm_dynamic_pe: 99.4 |
| NVIDIA_L4 | 4096 | 4096 | 1.13 | 1.36 | std::enable_if<T2>0&&_ty: 53.2 / void dynamic_scaled_int8: 88.7 | void rms_norm_dynamic_pe: 43.5 |
| NVIDIA_L4 | 4096 | 16384 | 1.08 | 1.08 | std::enable_if<T2>0&&_ty: 52.8 / void dynamic_scaled_int8: 41.8 | void rms_norm_dynamic_pe: 43.0 |

## Caveats
- Heuristic thresholds: {'slow': 1.1, 'fast': 0.9, 'dram_low': 60.0, 'scalar_B': 16.0, 'inst_ratio': 1.2}
- Clock setting: see logs/ncu_*.log (base on clusters; locked 2040 MHz + clock-control none on the L4 VM, as in Part 1).
- For V0, the L2 hit rate that matters for H2 is the quantization kernel's (it re-reads the normalized intermediate).
- Low DRAM %peak alone does not prove scalar loads (see proposal H1); confirm with stalls and the load-width columns.
- Warm mode means 'not flushed'; trust the L2 hit rate, not the label.
- Per-kernel detail: tables/memory_table.md and tables/execution_table.md.
