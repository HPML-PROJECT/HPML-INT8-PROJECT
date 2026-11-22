| variant | gpu | hidden | tokens | cache | T | H | kernel | regs/thr | block | grid | occ_ach_% | occ_theo_% | inst | gld_inst | ld_sectors/req | approx_B/ld_inst | top_stalls |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| v0 | NVIDIA_L4 | 2048 | 8 | cold | 8 | 2048 | TOTAL | nan | nan | nan | nan | nan | 30024.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 2048 | 8 | cold | 8 | 2048 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 1024.0 | 8.0 | 57.8 | 66.7 | 20312.0 | 320.0 | 16.0 | 512.0 | barrier=13.20, imc_miss=9.10, no_instruction=5.24 |
| v0 | NVIDIA_L4 | 2048 | 8 | cold | 8 | 2048 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 8.0 | 15.1 | 100.0 | 9712.0 | 128.0 | 32.0 | 1024.0 | imc_miss=5.87, barrier=3.72, long_scoreboard=2.77 |
| v1 | NVIDIA_L4 | 2048 | 8 | cold | 8 | 2048 | TOTAL | nan | nan | nan | nan | nan | 50760.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 2048 | 8 | cold | 8 | 2048 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 8.0 | 63.9 | 66.7 | 50760.0 | 1032.0 | 8.0 | 254.0 | barrier=8.96, imc_miss=6.13, no_instruction=4.00 |
| v0 | NVIDIA_L4 | 2048 | 256 | cold | 256 | 2048 | TOTAL | nan | nan | nan | nan | nan | 698112.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 2048 | 256 | cold | 256 | 2048 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 256.0 | 69.3 | 100.0 | 387328.0 | 10240.0 | 16.0 | 512.0 | long_scoreboard=69.15, barrier=10.50, imc_miss=3.81 |
| v0 | NVIDIA_L4 | 2048 | 256 | cold | 256 | 2048 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 256.0 | 57.7 | 100.0 | 310784.0 | 4096.0 | 32.0 | 1024.0 | barrier=22.14, long_scoreboard=15.95, imc_miss=5.14 |
| v1 | NVIDIA_L4 | 2048 | 256 | cold | 256 | 2048 | TOTAL | nan | nan | nan | nan | nan | 1624320.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 2048 | 256 | cold | 256 | 2048 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 256.0 | 62.8 | 66.7 | 1624320.0 | 33024.0 | 8.0 | 254.0 | barrier=10.21, long_scoreboard=3.95, wait=2.02 |
| v0 | NVIDIA_L4 | 2048 | 1024 | cold | 1024 | 2048 | TOTAL | nan | nan | nan | nan | nan | 2792448.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 2048 | 1024 | cold | 1024 | 2048 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 1024.0 | 90.4 | 100.0 | 1549312.0 | 40960.0 | 16.0 | 512.0 | long_scoreboard=72.79, barrier=11.87, short_scoreboard=6.93 |
| v0 | NVIDIA_L4 | 2048 | 1024 | cold | 1024 | 2048 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 1024.0 | 86.3 | 100.0 | 1243136.0 | 16384.0 | 32.0 | 1024.0 | barrier=29.09, long_scoreboard=24.54, wait=2.34 |
| v1 | NVIDIA_L4 | 2048 | 1024 | cold | 1024 | 2048 | TOTAL | nan | nan | nan | nan | nan | 6497280.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 2048 | 1024 | cold | 1024 | 2048 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 1024.0 | 62.0 | 66.7 | 6497280.0 | 132096.0 | 8.0 | 254.0 | barrier=8.12, long_scoreboard=2.60, wait=2.03 |
| v0 | NVIDIA_L4 | 2048 | 4096 | cold | 4096 | 2048 | TOTAL | nan | nan | nan | nan | nan | 11169792.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 2048 | 4096 | cold | 4096 | 2048 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 4096.0 | 90.5 | 100.0 | 6197248.0 | 163840.0 | 16.0 | 512.0 | long_scoreboard=75.40, short_scoreboard=13.21, barrier=12.69 |
| v0 | NVIDIA_L4 | 2048 | 4096 | cold | 4096 | 2048 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 4096.0 | 91.9 | 100.0 | 4972544.0 | 65536.0 | 32.0 | 1024.0 | barrier=31.73, long_scoreboard=26.38, wait=2.37 |
| v1 | NVIDIA_L4 | 2048 | 4096 | cold | 4096 | 2048 | TOTAL | nan | nan | nan | nan | nan | 25989120.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 2048 | 4096 | cold | 4096 | 2048 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 4096.0 | 60.7 | 66.7 | 25989120.0 | 528384.0 | 8.0 | 254.0 | barrier=8.12, long_scoreboard=2.54, wait=2.04 |
| v0 | NVIDIA_L4 | 2048 | 16384 | cold | 16384 | 2048 | TOTAL | nan | nan | nan | nan | nan | 44679168.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 2048 | 16384 | cold | 16384 | 2048 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 16384.0 | 93.7 | 100.0 | 24788992.0 | 655360.0 | 16.0 | 512.0 | long_scoreboard=96.88, barrier=33.46, short_scoreboard=30.56 |
| v0 | NVIDIA_L4 | 2048 | 16384 | cold | 16384 | 2048 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 16384.0 | 92.0 | 100.0 | 19890176.0 | 262144.0 | 32.0 | 1024.0 | barrier=43.84, long_scoreboard=29.60, wait=2.38 |
| v1 | NVIDIA_L4 | 2048 | 16384 | cold | 16384 | 2048 | TOTAL | nan | nan | nan | nan | nan | 103956480.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 2048 | 16384 | cold | 16384 | 2048 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 16384.0 | 62.9 | 66.7 | 103956480.0 | 2113536.0 | 8.0 | 254.0 | barrier=12.13, long_scoreboard=4.85, wait=2.04 |
| v0 | NVIDIA_L4 | 2048 | 8 | warm | 8 | 2048 | TOTAL | nan | nan | nan | nan | nan | 30024.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 2048 | 8 | warm | 8 | 2048 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 1024.0 | 8.0 | 48.5 | 66.7 | 20312.0 | 320.0 | 16.0 | 512.0 | barrier=7.76, imc_miss=5.78, short_scoreboard=2.54 |
| v0 | NVIDIA_L4 | 2048 | 8 | warm | 8 | 2048 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 8.0 | 13.3 | 100.0 | 9712.0 | 128.0 | 32.0 | 1024.0 | imc_miss=3.37, barrier=2.40, wait=2.23 |
| v1 | NVIDIA_L4 | 2048 | 8 | warm | 8 | 2048 | TOTAL | nan | nan | nan | nan | nan | 50760.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 2048 | 8 | warm | 8 | 2048 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 8.0 | 55.9 | 66.7 | 50760.0 | 1032.0 | 8.0 | 254.0 | barrier=6.03, imc_miss=2.48, wait=1.95 |
| v0 | NVIDIA_L4 | 2048 | 256 | warm | 256 | 2048 | TOTAL | nan | nan | nan | nan | nan | 698112.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 2048 | 256 | warm | 256 | 2048 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 256.0 | 61.5 | 100.0 | 387328.0 | 10240.0 | 16.0 | 512.0 | long_scoreboard=8.05, barrier=3.10, imc_miss=2.13 |
| v0 | NVIDIA_L4 | 2048 | 256 | warm | 256 | 2048 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 256.0 | 56.1 | 100.0 | 310784.0 | 4096.0 | 32.0 | 1024.0 | barrier=4.84, long_scoreboard=2.70, imc_miss=2.46 |
| v1 | NVIDIA_L4 | 2048 | 256 | warm | 256 | 2048 | TOTAL | nan | nan | nan | nan | nan | 1624320.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 2048 | 256 | warm | 256 | 2048 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 256.0 | 60.6 | 66.7 | 1624320.0 | 33024.0 | 8.0 | 254.0 | barrier=6.31, wait=2.02, short_scoreboard=1.97 |
| v0 | NVIDIA_L4 | 2048 | 1024 | warm | 1024 | 2048 | TOTAL | nan | nan | nan | nan | nan | 2792448.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 2048 | 1024 | warm | 1024 | 2048 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 1024.0 | 86.7 | 100.0 | 1549312.0 | 40960.0 | 16.0 | 512.0 | long_scoreboard=8.03, barrier=4.26, short_scoreboard=2.88 |
| v0 | NVIDIA_L4 | 2048 | 1024 | warm | 1024 | 2048 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 1024.0 | 75.5 | 100.0 | 1243136.0 | 16384.0 | 32.0 | 1024.0 | barrier=4.37, wait=2.39, long_scoreboard=2.03 |
| v1 | NVIDIA_L4 | 2048 | 1024 | warm | 1024 | 2048 | TOTAL | nan | nan | nan | nan | nan | 6497280.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 2048 | 1024 | warm | 1024 | 2048 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 1024.0 | 61.2 | 66.7 | 6497280.0 | 132096.0 | 8.0 | 254.0 | barrier=6.15, wait=2.04, short_scoreboard=2.01 |
| v0 | NVIDIA_L4 | 2048 | 4096 | warm | 4096 | 2048 | TOTAL | nan | nan | nan | nan | nan | 11169792.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 2048 | 4096 | warm | 4096 | 2048 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 4096.0 | 94.9 | 100.0 | 6197248.0 | 163840.0 | 16.0 | 512.0 | long_scoreboard=33.69, barrier=12.03, short_scoreboard=8.29 |
| v0 | NVIDIA_L4 | 2048 | 4096 | warm | 4096 | 2048 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 4096.0 | 81.2 | 100.0 | 4972544.0 | 65536.0 | 32.0 | 1024.0 | barrier=3.91, wait=2.43, not_selected=1.68 |
| v1 | NVIDIA_L4 | 2048 | 4096 | warm | 4096 | 2048 | TOTAL | nan | nan | nan | nan | nan | 25989120.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 2048 | 4096 | warm | 4096 | 2048 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 4096.0 | 60.4 | 66.7 | 25989120.0 | 528384.0 | 8.0 | 254.0 | barrier=8.00, long_scoreboard=2.31, wait=2.04 |
| v0 | NVIDIA_L4 | 2048 | 16384 | warm | 16384 | 2048 | TOTAL | nan | nan | nan | nan | nan | 44679168.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 2048 | 16384 | warm | 16384 | 2048 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 16384.0 | 96.4 | 100.0 | 24788992.0 | 655360.0 | 16.0 | 512.0 | long_scoreboard=100.87, barrier=49.95, short_scoreboard=35.20 |
| v0 | NVIDIA_L4 | 2048 | 16384 | warm | 16384 | 2048 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 16384.0 | 91.2 | 100.0 | 19890176.0 | 262144.0 | 32.0 | 1024.0 | barrier=63.07, long_scoreboard=34.13, short_scoreboard=3.67 |
| v1 | NVIDIA_L4 | 2048 | 16384 | warm | 16384 | 2048 | TOTAL | nan | nan | nan | nan | nan | 103956480.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 2048 | 16384 | warm | 16384 | 2048 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 16384.0 | 64.6 | 66.7 | 103956480.0 | 2113536.0 | 8.0 | 254.0 | barrier=14.33, long_scoreboard=5.81, wait=2.04 |
| v0 | NVIDIA_L4 | 4096 | 8 | cold | 8 | 4096 | TOTAL | nan | nan | nan | nan | nan | 41832.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 4096 | 8 | cold | 8 | 4096 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 1024.0 | 8.0 | 57.2 | 66.7 | 28376.0 | 640.0 | 16.0 | 512.0 | barrier=10.14, long_scoreboard=7.35, imc_miss=6.80 |
| v0 | NVIDIA_L4 | 4096 | 8 | cold | 8 | 4096 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 8.0 | 16.1 | 100.0 | 13456.0 | 256.0 | 32.0 | 1024.0 | imc_miss=5.14, long_scoreboard=4.78, wait=2.06 |
| v1 | NVIDIA_L4 | 4096 | 8 | cold | 8 | 4096 | TOTAL | nan | nan | nan | nan | nan | 69576.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 4096 | 8 | cold | 8 | 4096 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 8.0 | 65.8 | 66.7 | 69576.0 | 2056.0 | 8.0 | 255.0 | long_scoreboard=5.88, imc_miss=4.59, barrier=3.89 |
| v0 | NVIDIA_L4 | 4096 | 256 | cold | 256 | 4096 | TOTAL | nan | nan | nan | nan | nan | 1010432.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 4096 | 256 | cold | 256 | 4096 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 256.0 | 71.3 | 100.0 | 579840.0 | 20480.0 | 16.0 | 512.0 | long_scoreboard=102.39, barrier=7.25, imc_miss=2.88 |
| v0 | NVIDIA_L4 | 4096 | 256 | cold | 256 | 4096 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 256.0 | 58.8 | 100.0 | 430592.0 | 8192.0 | 32.0 | 1024.0 | long_scoreboard=42.36, barrier=10.63, imc_miss=3.72 |
| v1 | NVIDIA_L4 | 4096 | 256 | cold | 256 | 4096 | TOTAL | nan | nan | nan | nan | nan | 2226432.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 4096 | 256 | cold | 256 | 4096 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 256.0 | 65.4 | 66.7 | 2226432.0 | 65792.0 | 8.0 | 255.0 | long_scoreboard=10.76, barrier=6.80, not_selected=2.45 |
| v0 | NVIDIA_L4 | 4096 | 1024 | cold | 1024 | 4096 | TOTAL | nan | nan | nan | nan | nan | 4041728.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 4096 | 1024 | cold | 1024 | 4096 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 1024.0 | 90.8 | 100.0 | 2319360.0 | 81920.0 | 16.0 | 512.0 | long_scoreboard=107.95, barrier=10.10, lg_throttle=7.29 |
| v0 | NVIDIA_L4 | 4096 | 1024 | cold | 1024 | 4096 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 1024.0 | 86.5 | 100.0 | 1722368.0 | 32768.0 | 32.0 | 1024.0 | long_scoreboard=67.48, barrier=9.25, short_scoreboard=4.92 |
| v1 | NVIDIA_L4 | 4096 | 1024 | cold | 1024 | 4096 | TOTAL | nan | nan | nan | nan | nan | 8905728.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 4096 | 1024 | cold | 1024 | 4096 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 1024.0 | 65.2 | 66.7 | 8905728.0 | 263168.0 | 8.0 | 255.0 | long_scoreboard=8.01, barrier=5.16, not_selected=2.51 |
| v0 | NVIDIA_L4 | 4096 | 4096 | cold | 4096 | 4096 | TOTAL | nan | nan | nan | nan | nan | 16166912.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 4096 | 4096 | cold | 4096 | 4096 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 4096.0 | 89.9 | 100.0 | 9277440.0 | 327680.0 | 16.0 | 512.0 | long_scoreboard=136.92, barrier=28.06, short_scoreboard=19.79 |
| v0 | NVIDIA_L4 | 4096 | 4096 | cold | 4096 | 4096 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 4096.0 | 86.5 | 100.0 | 6889472.0 | 131072.0 | 32.0 | 1024.0 | long_scoreboard=70.98, barrier=10.05, short_scoreboard=4.75 |
| v1 | NVIDIA_L4 | 4096 | 4096 | cold | 4096 | 4096 | TOTAL | nan | nan | nan | nan | nan | 35622912.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 4096 | 4096 | cold | 4096 | 4096 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 4096.0 | 63.9 | 66.7 | 35622912.0 | 1052672.0 | 8.0 | 255.0 | long_scoreboard=12.65, barrier=10.41, not_selected=2.51 |
| v0 | NVIDIA_L4 | 4096 | 16384 | cold | 16384 | 4096 | TOTAL | nan | nan | nan | nan | nan | 64667648.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 4096 | 16384 | cold | 16384 | 4096 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 16384.0 | 91.8 | 100.0 | 37109760.0 | 1310720.0 | 16.0 | 512.0 | long_scoreboard=150.19, barrier=46.45, mio_throttle=35.51 |
| v0 | NVIDIA_L4 | 4096 | 16384 | cold | 16384 | 4096 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 16384.0 | 89.1 | 100.0 | 27557888.0 | 524288.0 | 32.0 | 1024.0 | long_scoreboard=85.53, barrier=26.10, short_scoreboard=7.26 |
| v1 | NVIDIA_L4 | 4096 | 16384 | cold | 16384 | 4096 | TOTAL | nan | nan | nan | nan | nan | 142491648.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 4096 | 16384 | cold | 16384 | 4096 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 16384.0 | 65.1 | 66.7 | 142491648.0 | 4210688.0 | 8.0 | 255.0 | barrier=21.55, long_scoreboard=15.36, not_selected=2.50 |
| v0 | NVIDIA_L4 | 4096 | 8 | warm | 8 | 4096 | TOTAL | nan | nan | nan | nan | nan | 41832.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 4096 | 8 | warm | 8 | 4096 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 1024.0 | 8.0 | 51.6 | 66.7 | 28376.0 | 640.0 | 16.0 | 512.0 | barrier=6.56, imc_miss=4.32, long_scoreboard=3.69 |
| v0 | NVIDIA_L4 | 4096 | 8 | warm | 8 | 4096 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 8.0 | 14.3 | 100.0 | 13456.0 | 256.0 | 32.0 | 1024.0 | long_scoreboard=2.51, imc_miss=2.49, wait=2.06 |
| v1 | NVIDIA_L4 | 4096 | 8 | warm | 8 | 4096 | TOTAL | nan | nan | nan | nan | nan | 69576.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 4096 | 8 | warm | 8 | 4096 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 8.0 | 59.2 | 66.7 | 69576.0 | 2056.0 | 8.0 | 255.0 | barrier=3.64, long_scoreboard=2.74, not_selected=2.28 |
| v0 | NVIDIA_L4 | 4096 | 256 | warm | 256 | 4096 | TOTAL | nan | nan | nan | nan | nan | 1010432.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 4096 | 256 | warm | 256 | 4096 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 256.0 | 64.5 | 100.0 | 579840.0 | 20480.0 | 16.0 | 512.0 | long_scoreboard=14.14, barrier=2.01, imc_miss=1.47 |
| v0 | NVIDIA_L4 | 4096 | 256 | warm | 256 | 4096 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 256.0 | 53.7 | 100.0 | 430592.0 | 8192.0 | 32.0 | 1024.0 | long_scoreboard=6.29, barrier=2.37, wait=2.11 |
| v1 | NVIDIA_L4 | 4096 | 256 | warm | 256 | 4096 | TOTAL | nan | nan | nan | nan | nan | 2226432.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 4096 | 256 | warm | 256 | 4096 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 256.0 | 62.2 | 66.7 | 2226432.0 | 65792.0 | 8.0 | 255.0 | barrier=3.75, long_scoreboard=2.61, not_selected=2.50 |
| v0 | NVIDIA_L4 | 4096 | 1024 | warm | 1024 | 4096 | TOTAL | nan | nan | nan | nan | nan | 4041728.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 4096 | 1024 | warm | 1024 | 4096 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 1024.0 | 86.5 | 100.0 | 2319360.0 | 81920.0 | 16.0 | 512.0 | long_scoreboard=14.20, barrier=3.78, short_scoreboard=2.68 |
| v0 | NVIDIA_L4 | 4096 | 1024 | warm | 1024 | 4096 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 1024.0 | 80.3 | 100.0 | 1722368.0 | 32768.0 | 32.0 | 1024.0 | long_scoreboard=4.61, barrier=2.48, wait=2.18 |
| v1 | NVIDIA_L4 | 4096 | 1024 | warm | 1024 | 4096 | TOTAL | nan | nan | nan | nan | nan | 8905728.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 4096 | 1024 | warm | 1024 | 4096 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 1024.0 | 62.9 | 66.7 | 8905728.0 | 263168.0 | 8.0 | 255.0 | barrier=3.68, not_selected=2.56, long_scoreboard=2.18 |
| v0 | NVIDIA_L4 | 4096 | 4096 | warm | 4096 | 4096 | TOTAL | nan | nan | nan | nan | nan | 16166912.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 4096 | 4096 | warm | 4096 | 4096 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 4096.0 | 93.6 | 100.0 | 9277440.0 | 327680.0 | 16.0 | 512.0 | long_scoreboard=153.95, barrier=42.73, mio_throttle=34.38 |
| v0 | NVIDIA_L4 | 4096 | 4096 | warm | 4096 | 4096 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 4096.0 | 52.4 | 100.0 | 6889472.0 | 131072.0 | 32.0 | 1024.0 | long_scoreboard=11.61, barrier=8.97, wait=2.19 |
| v1 | NVIDIA_L4 | 4096 | 4096 | warm | 4096 | 4096 | TOTAL | nan | nan | nan | nan | nan | 35622912.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 4096 | 4096 | warm | 4096 | 4096 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 4096.0 | 66.5 | 66.7 | 35622912.0 | 1052672.0 | 8.0 | 255.0 | barrier=22.12, long_scoreboard=15.64, not_selected=2.49 |
| v0 | NVIDIA_L4 | 4096 | 16384 | warm | 16384 | 4096 | TOTAL | nan | nan | nan | nan | nan | 64667648.0 | nan | nan | nan |  |
| v0 | NVIDIA_L4 | 4096 | 16384 | warm | 16384 | 4096 | std::enable_if<T2>0&&_typeConvert<T1>::exists, void>::type f | 30.0 | 256.0 | 16384.0 | 93.0 | 100.0 | 37109760.0 | 1310720.0 | 16.0 | 512.0 | long_scoreboard=150.88, barrier=52.98, mio_throttle=39.20 |
| v0 | NVIDIA_L4 | 4096 | 16384 | warm | 16384 | 4096 | void dynamic_scaled_int8_quant_kernel<BFloat16, float> | 36.0 | 256.0 | 16384.0 | 87.7 | 100.0 | 27557888.0 | 524288.0 | 32.0 | 1024.0 | long_scoreboard=67.96, barrier=56.36, short_scoreboard=9.34 |
| v1 | NVIDIA_L4 | 4096 | 16384 | warm | 16384 | 4096 | TOTAL | nan | nan | nan | nan | nan | 142491648.0 | nan | nan | nan |  |
| v1 | NVIDIA_L4 | 4096 | 16384 | warm | 16384 | 4096 | void rms_norm_dynamic_per_token_quant_kernel<BFloat16, signe | 32.0 | 1024.0 | 16384.0 | 66.3 | 66.7 | 142491648.0 | 4210688.0 | 8.0 | 255.0 | barrier=26.40, long_scoreboard=14.38, not_selected=2.50 |
