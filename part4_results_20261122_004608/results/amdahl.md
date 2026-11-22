# Amdahl bound (w8a8, mode=graphs, GPU=NVIDIA L4)

f = fraction of time in the accelerated region; S = 1 / ((1-f) + f/s). Bandwidth-model bound s = 11/7 = 1.57 (BF16 add+norm+quant vs fused).


## prefill_8x512
GPU busy fraction of wall time: 0.94

| basis | f | s=1.25 | s=1.5 | s=2 | s=3 | s=1.571 | s=inf |
|---|---|---|---|---|---|---|---|
| f_all_gpu | 15.21% | 1.0314x | 1.0534x | 1.0823x | 1.1128x | 1.0585x | 1.1794x |
| f_fusable_gpu | 7.57% | 1.0154x | 1.0259x | 1.0394x | 1.0532x | 1.0283x | 1.0819x |
| f_all_wall | 14.30% | 1.0294x | 1.0500x | 1.0770x | 1.1054x | 1.0548x | 1.1668x |
| f_fusable_wall | 7.12% | 1.0144x | 1.0243x | 1.0369x | 1.0498x | 1.0266x | 1.0766x |

## decode_b1
GPU busy fraction of wall time: 0.99

| basis | f | s=1.25 | s=1.5 | s=2 | s=3 | s=1.571 | s=inf |
|---|---|---|---|---|---|---|---|
| f_all_gpu | 6.52% | 1.0132x | 1.0222x | 1.0337x | 1.0454x | 1.0243x | 1.0697x |
| f_fusable_gpu | 2.92% | 1.0059x | 1.0098x | 1.0148x | 1.0199x | 1.0107x | 1.0301x |
| f_all_wall | 6.45% | 1.0131x | 1.0220x | 1.0333x | 1.0449x | 1.0240x | 1.0689x |
| f_fusable_wall | 2.89% | 1.0058x | 1.0097x | 1.0147x | 1.0197x | 1.0106x | 1.0298x |

## decode_b8
GPU busy fraction of wall time: 0.99

| basis | f | s=1.25 | s=1.5 | s=2 | s=3 | s=1.571 | s=inf |
|---|---|---|---|---|---|---|---|
| f_all_gpu | 6.24% | 1.0126x | 1.0212x | 1.0322x | 1.0434x | 1.0232x | 1.0665x |
| f_fusable_gpu | 2.91% | 1.0058x | 1.0098x | 1.0147x | 1.0198x | 1.0107x | 1.0299x |
| f_all_wall | 6.20% | 1.0126x | 1.0211x | 1.0320x | 1.0431x | 1.0231x | 1.0661x |
| f_fusable_wall | 2.89% | 1.0058x | 1.0097x | 1.0147x | 1.0197x | 1.0106x | 1.0298x |

## decode_b64
GPU busy fraction of wall time: 1.00

| basis | f | s=1.25 | s=1.5 | s=2 | s=3 | s=1.571 | s=inf |
|---|---|---|---|---|---|---|---|
| f_all_gpu | 4.53% | 1.0091x | 1.0153x | 1.0232x | 1.0311x | 1.0167x | 1.0474x |
| f_fusable_gpu | 2.11% | 1.0042x | 1.0071x | 1.0107x | 1.0143x | 1.0077x | 1.0216x |
| f_all_wall | 4.51% | 1.0091x | 1.0153x | 1.0231x | 1.0310x | 1.0167x | 1.0472x |
| f_fusable_wall | 2.10% | 1.0042x | 1.0071x | 1.0106x | 1.0142x | 1.0077x | 1.0215x |

Notes: *_gpu uses GPU kernel time (ignores CPU gaps); *_wall divides by profiled wall time (profiler overhead inflates it, so wall-based f is a lower bound). A kernel-level speedup is never an end-to-end speedup; use f_fusable for a realistic bound.
