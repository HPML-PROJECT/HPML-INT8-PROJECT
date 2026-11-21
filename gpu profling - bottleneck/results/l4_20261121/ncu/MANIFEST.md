# Nsight Compute reports (not stored in git)

The 40 `.ncu-rep` files from the L4 run (2026-11-21) total about 750 MB, with single files up to 93 MB, so they are kept out of git (GitHub's per-file limit is 100 MB).
They are in the run bundle `ws2_results_NVIDIA_L4_20261121_225219.tar.gz` under `reports/ncu/`. The tables in `../tables/` were generated from exactly these files.

Naming: `<variant>_<gpu>_H<hidden>_T<tokens>_<cold|warm>.ncu-rep`. Open them in the Nsight Compute GUI. To check a downloaded copy, compare the output of `sha256sum <file>` with the table below.

| File | Size (MB) | SHA-256 |
|---|---|---|
| v0_NVIDIA_L4_H2048_T1024_cold.ncu-rep | 10.2 | 8fce8b41ac77e19a95b2d7a0b770abddef47ced10cd553f94e0d486725dc7a42 |
| v0_NVIDIA_L4_H2048_T1024_warm.ncu-rep | 9.1 | abf55c2056e7e34435fd731b06556a2f3885cf85e5528bbcfa523c516f23fe34 |
| v0_NVIDIA_L4_H2048_T16384_cold.ncu-rep | 43.2 | d6c2e114b18a5efec40ff0d5821443974a1b044900a7c75a86adc19a163acaf2 |
| v0_NVIDIA_L4_H2048_T16384_warm.ncu-rep | 48.9 | 0a7b6d79b43dd693733dc6cf033142af6ff9ad92cd5e33fb6ea5c31a0c34f979 |
| v0_NVIDIA_L4_H2048_T256_cold.ncu-rep | 9.1 | 6f980b20d47535a54304fb24e4f8dfdc725554d84aa1b45024329bf3ddb85a6d |
| v0_NVIDIA_L4_H2048_T256_warm.ncu-rep | 8.8 | 8d6058e4683744d5a7fa5a38c940d7fd0fac71a66cdfabe865612c18e378b6e9 |
| v0_NVIDIA_L4_H2048_T4096_cold.ncu-rep | 14.5 | 7d0401907284ff0658794a8b90244b5ecd7c056297cd2780f582fcdc13590e16 |
| v0_NVIDIA_L4_H2048_T4096_warm.ncu-rep | 11.6 | 48da1c9c131b0895afa61760cd4a7ecdfc4be1aecbfc118efe08e9ea7cc9d8f0 |
| v0_NVIDIA_L4_H2048_T8_cold.ncu-rep | 7.6 | 7dd2cadd3303c2c2291a6c00ee5a4e4cb4da9ff2641f3c4bac24c19b249f1fa4 |
| v0_NVIDIA_L4_H2048_T8_warm.ncu-rep | 7.6 | af7ff442df75ea28e94e2144113030375e52f38ab6be16f695f83323cd6b07d1 |
| v0_NVIDIA_L4_H4096_T1024_cold.ncu-rep | 11.4 | db25b52321a724986eb5399d5e7d46b7a1c2df316995970c0c802d5f9a3ae54d |
| v0_NVIDIA_L4_H4096_T1024_warm.ncu-rep | 9.5 | 95cc8a362fdc513eafc9e386f199944b60804f8a9640b6155d97bc3fa3c056f3 |
| v0_NVIDIA_L4_H4096_T16384_cold.ncu-rep | 83.1 | 902d61c0ab039cc54fd06d8e7872e7805fb1c5c2b1df1be95547827a68a91627 |
| v0_NVIDIA_L4_H4096_T16384_warm.ncu-rep | 89.0 | fb3fbaee982884081851f7d4374f9431c7b427bdef92267e54a1323200d27d0b |
| v0_NVIDIA_L4_H4096_T256_cold.ncu-rep | 9.4 | b8e3db4900ba455a7e6caea5e89fdcd1b8dc99cee727b275d276f0e93390e072 |
| v0_NVIDIA_L4_H4096_T256_warm.ncu-rep | 8.8 | bf6cc6a697cf60fdcf1467c019c27f5ca1f3d30fc930eff234210996575bd122 |
| v0_NVIDIA_L4_H4096_T4096_cold.ncu-rep | 23.7 | 2c5650593f06d23971acff492abbf8f5709542d3b0af64a9045911d35249a92a |
| v0_NVIDIA_L4_H4096_T4096_warm.ncu-rep | 26.3 | 332494e55ad4051c01bb67be5fb3567a0ac75cef0134fbfb69da89eb709fd302 |
| v0_NVIDIA_L4_H4096_T8_cold.ncu-rep | 7.6 | 41a6f40c489ae7e94114aeed7b3dc7684175e9bec7c2d4e6c160aa0e8fb77a26 |
| v0_NVIDIA_L4_H4096_T8_warm.ncu-rep | 7.6 | 97a5f5b27807fdf34f99cb39efc8d3222a150ec35936605b248efacab13aea0f |
| v1_NVIDIA_L4_H2048_T1024_cold.ncu-rep | 7.3 | 89a5a57f677966e2c3bb0403f2f1d221d37a990e4560c09fa568e9998cb4f0b8 |
| v1_NVIDIA_L4_H2048_T1024_warm.ncu-rep | 7.2 | 5d7ac26975299c722ac584fe061926c31de8bca8cdd624cee33023a09b6e18d3 |
| v1_NVIDIA_L4_H2048_T16384_cold.ncu-rep | 27.7 | 3f2a3eb43f62caa928fb23e0f2ef3e225abaceb8ecad87f4e4aa0a10734084a8 |
| v1_NVIDIA_L4_H2048_T16384_warm.ncu-rep | 29.4 | b563fc01a5991d468c45312f566e8cea6b133e46e837375e81d47adf66db9834 |
| v1_NVIDIA_L4_H2048_T256_cold.ncu-rep | 6.5 | 793484c11542e9ff33a3596093128b41842b5848c7d607acd2b0d320447e9474 |
| v1_NVIDIA_L4_H2048_T256_warm.ncu-rep | 6.4 | 314a51735ba1c0367a274112c09aa277871bb38c55d9947bbc7f427836d360f9 |
| v1_NVIDIA_L4_H2048_T4096_cold.ncu-rep | 10.7 | 6c2b745c29fcaee69eef0c9bd4d500d7d74c881c1bfd6848d92b94c994c83a97 |
| v1_NVIDIA_L4_H2048_T4096_warm.ncu-rep | 10.6 | 69a61e6f382e206a96c3c2d69e9b756a892a3a44aea2ac0e3c2da80a149488f9 |
| v1_NVIDIA_L4_H2048_T8_cold.ncu-rep | 5.5 | 414d55e5a3d403829feae5b6ae172575fe4f4cd71d79ebe9703e78f0c909fd08 |
| v1_NVIDIA_L4_H2048_T8_warm.ncu-rep | 5.4 | 6e6c6dd9a528aa861a4342eebaee76050a984a78d8dbb90951ae2b7942d8d33e |
| v1_NVIDIA_L4_H4096_T1024_cold.ncu-rep | 7.9 | 6c923430074fe93466495363868f6fd2fb3eee55b318bfdcfb211deb711a83e7 |
| v1_NVIDIA_L4_H4096_T1024_warm.ncu-rep | 7.4 | 37bcf18934ffd279335a7ee2f292104efbceb361e5699345762d4bfee28064c0 |
| v1_NVIDIA_L4_H4096_T16384_cold.ncu-rep | 54.8 | ed90d802bbcc41f699ca41324e888944b24beadaa1c0306c6190fc0e25a82839 |
| v1_NVIDIA_L4_H4096_T16384_warm.ncu-rep | 57.4 | d40286c01cf6a4050bb7d8c1c7a154ac19d6ed050e7a5fe1e87ea22dcff74a10 |
| v1_NVIDIA_L4_H4096_T256_cold.ncu-rep | 6.8 | c4b7e69a644229caa0b0a59e270561a1ae3de90980a18784811cb1d4b551245a |
| v1_NVIDIA_L4_H4096_T256_warm.ncu-rep | 6.5 | 718204fb6095775d85ac1d7a080d95009095209b555171ee4f6878ad507eef43 |
| v1_NVIDIA_L4_H4096_T4096_cold.ncu-rep | 15.3 | e2117c5c36c5b87a92ce647061d77b5ecb0fbefe64281735f22e0a9a852a2276 |
| v1_NVIDIA_L4_H4096_T4096_warm.ncu-rep | 18.4 | d9366346b926131cf1e22a19bebdef23d1c322dd36b839a71d95aa7a92e85273 |
| v1_NVIDIA_L4_H4096_T8_cold.ncu-rep | 5.5 | c11b0b69d526b5141d16c435328b5d0e87e066c3cbe1c43341c3d205660780c3 |
| v1_NVIDIA_L4_H4096_T8_warm.ncu-rep | 5.5 | fc4ceb9aec7f7452bb4478ea4954591585b87805df88abd6a20f9d8e6273afd6 |
