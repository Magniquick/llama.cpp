# KAT-Coder-V2.5-Dev — quantisation sweeps

Raw measurements and the inferences drawn from them, for choosing a bit allocation on `Kwaipilot/KAT-Coder-V2.5-Dev` (Qwen3.6-35B-A3B fine-tune, `qwen3_5_moe`, 40 layers, 256 experts top-8, hidden 2048, vocab 248320).

Measured on Kaggle against the **bf16** weights — the 70 GB checkpoint is cheap to stream there and only JSON comes back. Kernels: `kaggle-quant/kat_sweep_kernel.py`, `kaggle-quant/kat_sweep2_kernel.py`. Raw JSON alongside this file in `kaggle-quant/sweeps/`.

> **Objective note.** We optimise ACTIVE BYTES PER TOKEN, not file size. Routed experts are ~93% of the file but only 8/256 are read per token, so the dense path dominates what decode actually streams. Every public recipe surveyed (bartowski, APEX, OptiQ, AWQ) optimises file size instead, which is why their allocations differ from ours.


## RAW — weight quantisation error, Sweep 1 (8 sampled layers, 10 configs)

715 records, 75 tensors. Relative L2 of dequantise-requantise vs bf16, mean over layers.

| class       | i4_g128 | i4_g32 | i4_g64 | i8_pc | u4_g128 | u4_g32 | u4_g64 | u4_g64_mse | u4_pc  | u8_pc |
|-------------|---------|--------|--------|-------|---------|--------|--------|------------|--------|-------|
| lm_head     | 12.82%  | 10.16% | 11.51% | 1.03% | 10.71%  | 8.28%  | 9.49%  | 9.30%      | 16.00% | 0.94% |
| in_proj_qkv | 13.34%  | 10.45% | 11.90% | 1.05% | 11.04%  | 8.45%  | 9.74%  | 9.55%      | 16.32% | 0.96% |
| in_proj_z   | 13.38%  | 10.45% | 11.91% | 1.09% | 11.05%  | 8.44%  | 9.74%  | 9.54%      | 16.68% | 0.98% |
| out_proj    | 13.52%  | 10.61% | 12.08% | 1.33% | 11.21%  | 8.54%  | 9.88%  | 9.68%      | 19.89% | 1.19% |
| q_proj      | 13.11%  | 10.33% | 11.72% | 1.04% | 10.89%  | 8.39%  | 9.64%  | 9.45%      | 16.11% | 0.95% |
| o_proj      | 13.27%  | 10.49% | 11.90% | 1.22% | 11.06%  | 8.50%  | 9.79%  | 9.59%      | 18.77% | 1.12% |
| k_proj      | 14.65%  | 11.17% | 12.91% | 1.19% | 11.94%  | 8.80%  | 10.35% | 10.14%     | 18.39% | 1.08% |
| v_proj      | 16.69%  | 12.03% | 14.31% | 1.46% | 13.19%  | 9.15%  | 11.06% | 10.85%     | 21.75% | 1.32% |
| shared_gate | 14.42%  | 10.94% | 12.65% | 1.21% | 11.71%  | 8.66%  | 10.15% | 9.95%      | 18.40% | 1.09% |
| shared_up   | 13.99%  | 10.77% | 12.37% | 1.15% | 11.45%  | 8.61%  | 10.02% | 9.81%      | 17.63% | 1.04% |
| shared_down | 14.87%  | 11.20% | 13.00% | 1.05% | 12.03%  | 8.81%  | 10.38% | 10.17%     | 15.59% | 0.92% |
| router      | 19.26%  | 13.13% | 16.10% | 1.90% | 14.65%  | 9.69%  | 12.05% | 11.82%     | 27.89% | 1.66% |
| in_proj_a   | 14.85%  | 11.07% | 12.90% | 1.26% | 11.99%  | 8.72%  | 10.28% | 10.08%     | 19.50% | 1.15% |
| in_proj_b   | 15.68%  | 11.49% | 13.52% | 1.40% | 12.45%  | 8.87%  | 10.59% | 10.38%     | 21.03% | 1.24% |

## RAW — weight quantisation error, Sweep 2 (all 40 layers, 17 configs)

5967 records, 351 tensors. Relative L2 of dequantise-requantise vs bf16, mean over layers.

| class       | i4_g128 | i4_g256 | i4_g32 | i4_g64 | i8_pc | u4_g128 | u4_g128_mse | u4_g256 | u4_g256_mse | u4_g32 | u4_g32_mse | u4_g64 | u4_g64_mse | u4_pc  | u4_pc_mse | u8_g128 | u8_pc |
|-------------|---------|---------|--------|--------|-------|---------|-------------|---------|-------------|--------|------------|--------|------------|--------|-----------|---------|-------|
| lm_head     | 12.82%  | 14.18%  | 10.16% | 11.51% | 1.03% | 10.71%  | 10.29%      | 11.90%  | 11.09%      | 8.28%  | 8.25%      | 9.49%  | 9.29%      | 16.00% | 13.22%    | 0.63%   | 0.94% |
| in_proj_qkv | 13.39%  | 14.80%  | 10.52% | 11.97% | 1.05% | 11.11%  | 10.68%      | 12.42%  | 11.57%      | 8.50%  | 8.48%      | 9.81%  | 9.60%      | 16.32% | 13.53%    | 0.65%   | 0.96% |
| in_proj_z   | 13.27%  | 14.71%  | 10.43% | 11.86% | 1.06% | 11.01%  | 10.58%      | 12.30%  | 11.46%      | 8.44%  | 8.42%      | 9.73%  | 9.52%      | 16.33% | 13.53%    | 0.65%   | 0.96% |
| out_proj    | 13.09%  | 14.58%  | 10.39% | 11.76% | 1.29% | 10.93%  | 10.51%      | 12.28%  | 11.45%      | 8.43%  | 8.40%      | 9.69%  | 9.48%      | 19.35% | 15.89%    | 0.64%   | 1.16% |
| q_proj      | 12.96%  | 14.27%  | 10.31% | 11.65% | 1.00% | 10.83%  | 10.41%      | 12.03%  | 11.22%      | 8.39%  | 8.36%      | 9.62%  | 9.42%      | 15.65% | 12.96%    | 0.64%   | 0.92% |
| o_proj      | 12.92%  | 14.18%  | 10.30% | 11.63% | 1.15% | 10.82%  | 10.40%      | 12.00%  | 11.20%      | 8.38%  | 8.36%      | 9.62%  | 9.41%      | 17.89% | 14.65%    | 0.64%   | 1.06% |
| k_proj      | 14.08%  | 15.66%  | 10.88% | 12.48% | 1.12% | 11.56%  | 11.11%      | 13.03%  | 12.13%      | 8.67%  | 8.64%      | 10.11% | 9.89%      | 17.41% | 14.40%    | 0.68%   | 1.02% |
| v_proj      | 15.00%  | 16.80%  | 11.31% | 13.14% | 1.24% | 12.16%  | 11.69%      | 13.89%  | 12.95%      | 8.88%  | 8.86%      | 10.48% | 10.26%     | 18.83% | 15.70%    | 0.72%   | 1.13% |
| shared_gate | 15.05%  | 16.99%  | 11.30% | 13.15% | 1.28% | 12.15%  | 11.67%      | 13.90%  | 12.95%      | 8.85%  | 8.83%      | 10.46% | 10.24%     | 19.35% | 16.05%    | 0.72%   | 1.15% |
| shared_up   | 14.39%  | 16.09%  | 11.05% | 12.71% | 1.18% | 11.76%  | 11.30%      | 13.31%  | 12.39%      | 8.77%  | 8.75%      | 10.26% | 10.04%     | 18.07% | 14.95%    | 0.69%   | 1.07% |
| shared_down | 14.07%  | 15.74%  | 10.84% | 12.45% | 0.97% | 11.53%  | 11.08%      | 13.02%  | 12.13%      | 8.64%  | 8.62%      | 10.07% | 9.86%      | 14.60% | 13.03%    | 0.68%   | 0.86% |
| router      | 21.35%  | 25.34%  | 14.24% | 17.74% | 2.14% | 16.11%  | 15.49%      | 19.49%  | 18.14%      | 10.22% | 10.22%     | 12.99% | 12.73%     | 30.76% | 25.66%    | 0.95%   | 1.86% |
| in_proj_a   | 15.78%  | 18.20%  | 11.41% | 13.48% | 1.42% | 12.46%  | 11.96%      | 14.50%  | 13.46%      | 8.88%  | 8.86%      | 10.56% | 10.34%     | 21.50% | 17.68%    | 0.73%   | 1.26% |
| in_proj_b   | 17.13%  | 19.88%  | 12.15% | 14.55% | 1.50% | 13.39%  | 12.85%      | 15.78%  | 14.64%      | 9.18%  | 9.16%      | 11.16% | 10.93%     | 22.95% | 18.91%    | 0.79%   | 1.35% |

## RAW — activation-aware error (streaming kernel, one layer materialised at a time in real bf16)

`||X(W - Q(W))|| / ||XW||` using each Linear's REAL captured inputs from a code/reasoning calibration set. This is the metric that matters: plain weight error ignores what the layer actually sees.

| class       | i4_g128 | i4_g256 | i4_g32 | i4_g64 | i8_pc | u4_g128 | u4_g128_mse | u4_g256 | u4_g256_mse | u4_g32 | u4_g32_mse | u4_g64 | u4_g64_mse | u4_pc  | u4_pc_mse | u8_pc |
|-------------|---------|---------|--------|--------|-------|---------|-------------|---------|-------------|--------|------------|--------|------------|--------|-----------|-------|
| in_proj_qkv | 7.41%   | 8.16%   | 5.84%  | 6.63%  | 0.59% | 6.20%   | 6.66%       | 6.91%   | 7.40%       | 4.78%  | 5.23%      | 5.51%  | 5.68%      | 8.99%  | 8.30%     | 0.54% |
| in_proj_z   | 4.57%   | 5.06%   | 3.56%  | 4.07%  | 0.37% | 3.82%   | 4.65%       | 4.25%   | 5.33%       | 2.94%  | 3.69%      | 3.38%  | 3.76%      | 5.64%  | 5.65%     | 0.33% |
| out_proj    | 13.85%  | 15.36%  | 11.11% | 12.56% | 1.38% | 11.67%  | 11.39%      | 13.04%  | 12.71%      | 8.98%  | 9.04%      | 10.33% | 10.16%     | 19.61% | 17.40%    | 1.26% |
| q_proj      | 3.64%   | 3.99%   | 2.88%  | 3.27%  | 0.29% | 3.12%   | 4.12%       | 3.45%   | 4.93%       | 2.44%  | 3.18%      | 2.79%  | 3.20%      | 4.44%  | 5.12%     | 0.27% |
| o_proj      | 12.62%  | 13.83%  | 9.90%  | 11.28% | 1.10% | 10.47%  | 11.00%      | 11.66%  | 12.89%      | 7.99%  | 8.35%      | 9.29%  | 9.33%      | 17.18% | 16.11%    | 1.02% |
| k_proj      | 13.13%  | 14.46%  | 10.35% | 11.66% | 1.04% | 11.02%  | 10.91%      | 12.25%  | 12.20%      | 8.35%  | 8.48%      | 9.63%  | 9.57%      | 16.01% | 14.30%    | 0.96% |
| v_proj      | 14.76%  | 16.15%  | 11.69% | 13.21% | 1.39% | 12.43%  | 12.31%      | 13.81%  | 13.53%      | 9.56%  | 9.57%      | 11.11% | 10.93%     | 17.66% | 15.99%    | 1.26% |
| shared_gate | 7.98%   | 9.02%   | 5.99%  | 6.99%  | 0.67% | 6.44%   | 6.83%       | 7.39%   | 7.80%       | 4.71%  | 5.13%      | 5.57%  | 5.73%      | 10.30% | 9.23%     | 0.61% |
| shared_up   | 11.04%  | 12.32%  | 8.50%  | 9.80%  | 0.90% | 9.06%   | 8.94%       | 10.20%  | 9.90%       | 6.76%  | 6.90%      | 7.91%  | 7.86%      | 13.82% | 11.76%    | 0.82% |
| shared_down | 11.38%  | 12.63%  | 8.82%  | 10.08% | 0.79% | 9.45%   | 10.22%      | 10.59%  | 12.01%      | 7.16%  | 7.37%      | 8.31%  | 8.53%      | 11.85% | 14.91%    | 0.71% |
| in_proj_a   | 7.99%   | 9.11%   | 5.61%  | 6.81%  | 0.71% | 6.37%   | 7.42%       | 7.38%   | 8.82%       | 4.52%  | 5.12%      | 5.50%  | 5.79%      | 10.62% | 10.54%    | 0.64% |
| in_proj_b   | 6.05%   | 7.10%   | 4.16%  | 5.14%  | 0.54% | 4.85%   | 5.97%       | 5.75%   | 7.19%       | 3.37%  | 3.98%      | 4.06%  | 4.60%      | 8.45%  | 8.47%     | 0.49% |

## RAW — lm_head exact top-1 and KLD (streaming kernel, one layer materialised at a time in real bf16)

> **N = 159 tokens.** Every `top1_match` here is an exact k/159 — the calibration set was six hand-written prompts. The apparent spread between 4-bit configs is a handful of tokens, and no pairwise comparison in this table is significant. Wilson 95% intervals on the FLIP rate:

| cfg         | flips | flip rate | Wilson 95% CI  |
|-------------|-------|-----------|----------------|
| u8_pc       | 0/159 | 0.00%     | 0.00% – 2.36%  |
| i8_pc       | 1/159 | 0.63%     | 0.11% – 3.48%  |
| u4_g32_mse  | 2/159 | 1.26%     | 0.35% – 4.47%  |
| u4_g64_mse  | 4/159 | 2.52%     | 0.98% – 6.29%  |
| u4_g128_mse | 4/159 | 2.52%     | 0.98% – 6.29%  |
| u4_g32      | 5/159 | 3.14%     | 1.35% – 7.15%  |
| u4_g256_mse | 5/159 | 3.14%     | 1.35% – 7.15%  |
| i4_g32      | 6/159 | 3.77%     | 1.74% – 7.99%  |
| i4_g64      | 6/159 | 3.77%     | 1.74% – 7.99%  |
| u4_g64      | 7/159 | 4.40%     | 2.15% – 8.81%  |
| u4_g128     | 8/159 | 5.03%     | 2.57% – 9.61%  |
| i4_g128     | 8/159 | 5.03%     | 2.57% – 9.61%  |
| u4_g256     | 8/159 | 5.03%     | 2.57% – 9.61%  |
| i4_g256     | 8/159 | 5.03%     | 2.57% – 9.61%  |
| u4_pc       | 9/159 | 5.66%     | 3.01% – 10.41% |
| u4_pc_mse   | 9/159 | 5.66%     | 3.01% – 10.41% |

What IS supported despite that: the flip count falls monotonically as bits rise (9, 8, 8, 7, 5, 0), and a monotone trend across six independent configs is far stronger evidence than any single pair. "u8 lm_head beats u4 lm_head" is real; "u4_g32_mse beats u4_g32 by 3 tokens" is not measured. A re-measurement at ~1536 tokens is in flight.

Logits are `h @ W`, so with the real final hidden states captured these are EXACT, not estimated: KL against the bf16 reference distribution and top-1 agreement over the calibration positions.

| cfg         | eff bits | mean KLD | top-1 match | MB  |
|-------------|----------|----------|-------------|-----|
| u4_pc       | 4.010    | 0.01444  | 94.34%      | 255 |
| u4_pc_mse   | 4.010    | 0.01264  | 94.34%      | 255 |
| i4_g256     | 4.062    | 0.01204  | 94.97%      | 258 |
| u4_g256     | 4.078    | 0.00909  | 94.97%      | 259 |
| u4_g256_mse | 4.078    | 0.00757  | 96.86%      | 259 |
| i4_g128     | 4.125    | 0.00979  | 94.97%      | 262 |
| u4_g128     | 4.156    | 0.00744  | 94.97%      | 264 |
| u4_g128_mse | 4.156    | 0.00920  | 97.48%      | 264 |
| i4_g64      | 4.250    | 0.00935  | 96.23%      | 270 |
| u4_g64      | 4.312    | 0.00560  | 95.60%      | 274 |
| u4_g64_mse  | 4.312    | 0.00680  | 97.48%      | 274 |
| i4_g32      | 4.500    | 0.00637  | 96.23%      | 286 |
| u4_g32      | 4.625    | 0.00482  | 96.86%      | 294 |
| u4_g32_mse  | 4.625    | 0.00561  | 98.74%      | 294 |
| i8_pc       | 8.008    | 0.00006  | 99.37%      | 509 |
| u8_pc       | 8.012    | 0.00005  | 100.00%     | 509 |

## RAW — lm_head, REDONE (N=4096, reference bug fixed)

This supersedes the N=159 table above. Two things were wrong with that one: the sample was 159 tokens, and in the first re-run attempt the reference logits were recomputed with the SAME quantised head as the candidate, which cancelled the effect being measured (the giveaway was a bf16 row of exactly 0.00% for every config). Reference is now always the bf16 head.

| cfg        | flips    | flip rate | Wilson 95% CI | confident | mean KLD | p99 KLD |
|------------|----------|-----------|---------------|-----------|----------|---------|
| bf16       | 0/4096   | 0.00%     | 0.00% – 0.09% | 0         | 0.00000  | 0.00000 |
| u8_g128    | 10/4096  | 0.24%     | 0.13% – 0.45% | 0         | 0.00003  | 0.00021 |
| u8_pc      | 16/4096  | 0.39%     | 0.24% – 0.63% | 0         | 0.00005  | 0.00039 |
| u4_g32_mse | 108/4096 | 2.64%     | 2.19% – 3.17% | 0         | 0.00480  | 0.03438 |
| u4_g32     | 117/4096 | 2.86%     | 2.39% – 3.41% | 0         | 0.00440  | 0.03249 |
| u4_g64     | 141/4096 | 3.44%     | 2.93% – 4.05% | 0         | 0.00602  | 0.04474 |
| u4_g128    | 151/4096 | 3.69%     | 3.15% – 4.31% | 0         | 0.00727  | 0.05757 |

**u8 beats u4 decisively** (0.39% vs 2.86%, non-overlapping intervals). **The `_mse` advantage did not survive**: it looked like 1.26% vs 3.14% at N=159 and was recorded as "free, take it"; at N=4096 it is 2.64% vs 2.86% with heavily overlapping intervals, i.e. not measured. That was a 3-token artifact. Note `u4_g32` has the LOWER mean KLD while having MORE flips, so the KLD/top-1 tension is real but small. **Zero confident flips in every config** — even u4_g128's 151 flips all land on tokens bf16 was unsure of, which is the third independent confirmation of the near-tie bound.


## RAW — KAT vs Ornith, paired, at the DEPLOYED quant

Teacher-forced NLL of the CANONICAL SOLUTION given the prompt, over MBPP + HumanEval at a pinned commit. Scored in **bits per UTF-8 byte** because the two models do NOT share a tokenizer (both Qwen2Tokenizer, vocab files 12.8 MB vs 20.0 MB) — per-token loss would flatter whichever tokenises coarser. n is capped by COMPUTE, not data: one 40-layer streaming forward is ~20 min per 4096 tokens per arm.

| model  | arm  | n   | solution bytes | bits/byte |
|--------|------|-----|----------------|-----------|
| kat    | bf16 | 100 | 16007          | 0.2698    |
| kat    | opt1 | 100 | 16007          | 0.2715    |
| ornith | bf16 | 100 | 16007          | 0.2658    |
| ornith | opt1 | 100 | 16007          | 0.2683    |

**Paired result: KAT better on 50/100 items (50%), sign test p = 1.00.** There is NO evidence that KAT-Coder is the stronger coder on this metric; if anything Ornith is ahead by 0.0017 BPB, which is nothing. The model swap was premised on KAT being better and this does not support it.

**Quantisation is nearly free at the likelihood level**: bf16 -> opt1 costs +0.6% BPB on KAT and +0.9% on Ornith. Independent support for the allocation.

> Caveat, stated plainly: teacher-forced likelihood is NOT pass@1. A model can rank the reference solution well and still generate badly, and KAT is tagged `code`/`agent` so it may win on agentic work this cannot see. This rules out 'KAT is obviously better', not 'KAT is better'.


## RAW — does expert dominance REPLICATE on Ornith?

| track       | top-1 flips | flips | mean KLD |
|-------------|-------------|-------|----------|
| ref         | 0.00%       | 0     | 0.00000  |
| dense_u8pc  | 1.56%       | 24    | 0.00435  |
| dense_u4g64 | 7.88%       | 121   | 0.07353  |
| exp_u4g128  | 5.47%       | 84    | 0.04699  |
| exp_u4g64   | 5.53%       | 85    | 0.03005  |
| exp_u4g32   | 4.36%       | 67    | 0.02445  |

**Yes.** Experts at g64 cost 5.53% flips against 1.56% for the ENTIRE dense path at u8 — a 3.5x ratio, starker than KAT's 2.3x. The finding that redirected the whole allocation away from the activation-error proxy is not an artifact of one checkpoint.

**Cross-harness check also passes.** Deployed Ornith IRs measured KLD 0.1596 (u8 dense, arm B) vs 0.2313 (u4 dense, ABCD) against a Q8_0 reference; this harness independently ranks u8 dense (0.00435) below u4 dense (0.07353). Different reference, different harness, same ordering — the falsification test this run existed to perform.


---

# INFERRED

Derived from: sweep 1 weight error, sweep 2 weight error (all layers), activation-aware error, exact lm_head top-1/KLD.

## Pareto frontier — error vs EFFECTIVE bits (dense classes)

Effective bits include the per-group scale and zero-point, so a small group is not scored as free.

| cfg         | eff bits | mean rel-L2 | status     |
|-------------|----------|-------------|------------|
| u4_pc_mse   | 4.010    | 14.50%      | **PARETO** |
| u4_pc       | 4.010    | 17.35%      | dominated  |
| i4_g256     | 4.062    | 15.53%      | dominated  |
| u4_g256_mse | 4.078    | 12.03%      | **PARETO** |
| u4_g256     | 4.078    | 12.91%      | dominated  |
| i4_g128     | 4.125    | 13.93%      | dominated  |
| u4_g128_mse | 4.156    | 11.00%      | **PARETO** |
| u4_g128     | 4.156    | 11.45%      | dominated  |
| i4_g64      | 4.250    | 12.35%      | dominated  |
| u4_g64_mse  | 4.312    | 9.81%       | **PARETO** |
| u4_g64      | 4.312    | 10.02%      | dominated  |
| i4_g32      | 4.500    | 10.78%      | dominated  |
| u4_g32_mse  | 4.625    | 8.59%       | **PARETO** |
| u4_g32      | 4.625    | 8.62%       | dominated  |
| i8_pc       | 8.008    | 1.14%       | **PARETO** |
| u8_pc       | 8.012    | 1.03%       | **PARETO** |
| u8_g128     | 8.188    | 0.67%       | **PARETO** |

Frontier: `u4_pc_mse -> u4_g256_mse -> u4_g128_mse -> u4_g64_mse -> u4_g32_mse -> i8_pc -> u8_pc -> u8_g128`


## Measured sensitivity ranking — and where WEIGHT error misranks it

The allocation table below was originally ordered using weight error plus a sensitivity ordering borrowed from Unsloth's Qwen3.5-35B-A3B ablation. This section replaces that with KAT's own ACTIVATION-AWARE measurement and reports the disagreement explicitly, since a borrowed ordering from a sibling model is exactly the kind of assumption that should not survive contact with data.

| class       | act rank | act err @u4_g64 | wt rank | wt err @u4_g64 | MB @u4_g64 | verdict                     |
|-------------|----------|-----------------|---------|----------------|------------|-----------------------------|
| v_proj      | 1        | 11.11%          | 3       | 10.48%         | 6          | weight error UNDER-rates it |
| out_proj    | 2        | 10.33%          | 10      | 9.69%          | 136        | weight error UNDER-rates it |
| k_proj      | 3        | 9.63%           | 6       | 10.11%         | 6          | weight error UNDER-rates it |
| o_proj      | 4        | 9.29%           | 12      | 9.62%          | 45         | weight error UNDER-rates it |
| shared_down | 5        | 8.31%           | 7       | 10.07%         | 23         | weight error UNDER-rates it |
| shared_up   | 6        | 7.91%           | 5       | 10.26%         | 23         | agrees                      |
| shared_gate | 7        | 5.57%           | 4       | 10.46%         | 23         | weight error OVER-rates it  |
| in_proj_qkv | 8        | 5.51%           | 8       | 9.81%          | 271        | agrees                      |
| in_proj_a   | 9        | 5.50%           | 2       | 10.56%         | 1          | weight error OVER-rates it  |
| in_proj_b   | 10       | 4.06%           | 1       | 11.16%         | 1          | weight error OVER-rates it  |
| in_proj_z   | 11       | 3.38%           | 9       | 9.73%          | 136        | weight error OVER-rates it  |
| q_proj      | 12       | 2.79%           | 11      | 9.62%          | 90         | agrees                      |

Classes the two metrics disagree on by more than one place: `v_proj, out_proj, k_proj, o_proj, shared_down, shared_gate, in_proj_a, in_proj_b, in_proj_z`. Those are the entries where the allocation below should follow the activation-aware column.


## Where the measurement changes the story

**Coverage gap.** Activation-aware error covers only `nn.Linear` modules, because that is what a forward hook can capture inputs for. Absent: `router, expert_down`. The router is a `Qwen3_5MoeTopkRouter` and the routed experts are stacked Parameters, so neither is hooked — their entries in the allocation below still rest on WEIGHT error plus the argument that a mis-routed token is a discrete error, not on activation-aware data. `lm_head` sits outside the decoder layers and is measured exactly instead (top-1/KLD), which is strictly better.

**Correction to a stated reason.** The allocation table keeps `in_proj_a`/`in_proj_b` at f16 on the grounds that they are the "worst KLD-per-MB by an order of magnitude" — an ordering carried over from the Qwen3.5-35B-A3B ablation. On KAT they measure among the LEAST activation-sensitive classes (`in_proj_a` 9/12, `in_proj_b` 10/12). The RECOMMENDATION still stands, but for a different reason: both are ~1 MB, so holding them at f16 costs ~2 MB of active weight and is not worth relitigating. The sensitivity claim itself does not transfer to this model.

**KLD and top-1 disagree on `lm_head`.** Best KLD among 4-bit configs is `u4_g32` (0.00482 KLD, 96.855% top-1); best top-1 is `u4_g32_mse` (98.742% top-1, 0.00561 KLD). The MSE-shrink search minimises weight L2, which tightens the argmax but widens the tail of the distribution — so it wins on top-1 while LOSING on KLD. Pick by which one the deployment cares about: greedy decoding follows top-1, speculative drafting and any sampled decode follow the full distribution. For a drafted setup like ours, `u4_g32` is the right default.


## Recommended allocation for KAT

| tensor                   | choice          | why                                                                                                                                |
|--------------------------|-----------------|------------------------------------------------------------------------------------------------------------------------------------|
| lm_head                  | u4 g64 (mse)    | compress FIRST — lowest error of any class, 509 MB always-active. Contradicts bartowski `_L`/`_XL` and AWQ, which both protect it. |
| in_proj_qkv              | u4 g64 (mse)    | largest dense tensor, low sensitivity — best MB/error                                                                              |
| in_proj_z, q_proj        | u4 g64 (mse)    | low sensitivity                                                                                                                    |
| out_proj, o_proj         | u4 g32          | write into the residual stream; low WEIGHT error but high end-to-end sensitivity — trust activation-aware, not weight, error here  |
| k_proj, v_proj           | i8 (leave)      | 3rd/4th most sensitive per MB and only ~19 MB active; compressing buys ~0.3 t/s for ~26% of the quality loss                       |
| router (mlp.gate)        | f16             | MOST quantisation-sensitive tensor in the model (12.05% @g64, 27.9% per-channel); a wrong expert is a discrete error               |
| in_proj_a / in_proj_b    | f16             | worst KLD-per-MB by an order of magnitude; feed SoftPlus/Sigmoid into the GDN recurrence so error compounds through state          |
| layer 39 k/v/shared_down | i8 (hold)       | consistent outlier: 12.5-12.8% vs ~9.9% mid-stack                                                                                  |
| routed experts           | u4 g128 (leave) | execute f16 in `moe_3gemm_fused_compressed` — int8 would double active expert bytes for ZERO compute gain                          |
| embedding                | i8 (leave)      | a Gather: zero streamed bytes, and measures MORE sensitive than lm_head                                                            |

## Platform constraints that override any allocation

These were learned by breaking things and cost more time than the tuning:

1. **No extra nodes in the dequant chain.** The pattern must be exactly `MatMul <- Convert(f32) <- Reshape(f16) <- Multiply <- Subtract <- Convert <- Const(u4)`. One superfluous `Convert` breaks OV's pattern match, the plugin materialises f32 weights every forward, and throughput drops 2.1x (22 -> 13.5 t/s) with the file on disk still perfectly compressed and no error raised anywhere.

2. **Horizontally-fused tensors must share a group size.** q/k/v fuse into one `FullyConnectedCompressed` and their scales are concatenated; mismatched groups fail at compile time. Set the group by the SENSITIVE members (k/v), not the cheap one (q).

3. **Nothing below group 32** — `fully_connected_gpu_gemv.cl` has a hard `#error`.

3b. **8-bit asymmetric + grouped falls off the fast kernel: ~11x.** Measured on stacked dummy MatMuls (D=8192, L=4): u8 g128 asymmetric **41.68 ms** vs u8 per-channel 3.65 ms vs u4 g128 asymmetric 1.89 ms. It is not 8-bit and it is not grouping — **symmetric i8 g128 is fine at 4.85 ms** (+37% over per-channel, i.e. just its scale bytes). The broken combination is 8 bits AND a zero-point AND a group size. The runtime still reports these weights as compressed, so the only way to catch it is to time it. This matters because the end-to-end quality data RECOMMENDS grouped 8-bit dense (u8_g128 2.08% top-1 flips vs u8_pc 2.60%) — so the quality-optimal choice is the unshippable one, and the correct response is symmetric i8 with a group, not per-channel.

3c. **The MoE fused path is all-or-nothing u4.** `KeepMOE3GemmConstPrecision` matches only if EVERY expert constant is `u4` (3 routed weights + 3 zero-points, plus 3 shared-expert weights + 3 zero-points when a shared expert is present). One u8 constant, or symmetric u4 with no zero-point, and the pattern misses, `enable_keep_const_precision` is never set, and `ConvertPrecision` decompresses the weights at compile time. Measured cost of losing the compressed path: **4.2x**, and it is pure bandwidth (both arms run at 96-100 GB/s; only the byte count differs). This forecloses u8 experts AND u8 shared experts, the best-evidenced MoE quality win in the literature.

3d. **OpenVINO folds a `+1` into the decoder RMSNorm constants.** Measured by range-fetching the HF tensors and diffing against the donor IR: `input_layernorm` HF mean 0.0311 vs OV 1.0311, `post_attention_layernorm` -0.1049 vs 0.8951 — **exactly +1.0000 in both** — while `linear_attn.norm` matches at 0.0000. Qwen3.5/3.6 use zero-centred RMSNorm weights (`x * (1 + w)`) and the export bakes the +1 into the constant. It is NOT uniform across norms, which is what makes it dangerous: regenerate norms from HF without it and the model still runs, just wrong. Weight matrices are unaffected — dense classes match HF at cosine 0.995 (pure u4 error) with zero mean offset, so HF->OV needs no permutation or sign flip. The `vperm`/`t_nega` transforms in the Ornith transplant were GGUF-specific, not OV-specific.

4. **g32 may cost prefill** (not decode): the `bf_tiled` kernel branches on `TILE_IFM_ELEMENTS_SIZE > DECOMPRESSION_SCALE_GROUP_SIZE`. Re-measure prefill after any g32 change.


## What the public releases do differently

| release                   | allocation                                                                          | vs us                                                                                                                       |
|---------------------------|-------------------------------------------------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------|
| bartowski GGUF `_L`/`_XL` | Q8_0 for embed+output                                                               | **opposite of ours** on lm_head                                                                                             |
| mudler APEX               | layer gradient, edge layers richer; experts hardest                                 | partly agrees: our depth data shows 27-31% spread for v_proj/shared_down/in_proj_b but only 2-11% for the big dense tensors |
| mlx OptiQ 4bit            | 400 proj @4-bit, 111 @8-bit, ~4.51 bpw                                              | mixed, similar spirit                                                                                                       |
| cyankiwi AWQ INT4         | g32, and the ENTIRE dense path left unquantised (272 ignored modules incl. lm_head) | **opposite of ours** — correct for file size, worst possible for active bytes                                               |

## Caveats

* Weight error is a PROXY and is known to misrank this architecture (`out_proj` has low weight error, high sensitivity). Prefer the activation-aware table where both exist.

* Only `lm_head` has exact top-1/KLD. Everything else needs the end-to-end KLD harness (`kaggle-quant/kld_score.py`) against a model-specific reference — the 0.1596 baseline is Ornith's and does NOT transfer to KAT.

* Calibration for the activation-aware pass is a small code/reasoning set; a different domain could reorder the middle of the table.

