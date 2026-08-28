# LLM inference performance on the Yoga Pro 7 14IAH10 (Arrow Lake, Arc 140T iGPU)

Everything measured about prefill, decode, XMX, and the theoretical ceilings, for
Ornith-1.0-9B / -35B on OpenVINO and llama.cpp. Sessions of 2026-07-28 → 2026-08-01.

Every number is tagged **[M]** measured, **[D]** derived from measured numbers, or
**[R]** retracted/wrong. Retracted claims are kept deliberately — several were
believed for hours and re-derived from them, so knowing they are dead matters.

---

## 1. Hardware ceilings

### 1.1 Memory bandwidth — the real ceiling is ~106 GB/s, not 136

| quantity | value | source |
|---|---|---|
| theoretical LPDDR5x-8533, 128-bit | 136.5 GB/s | spec |
| **achievable global read, Vulkan** | **105.9 GB/s** | [M] clpeak |
| achievable global read, OpenCL/NEO | 102.4 GB/s | [M] clpeak |
| local memory (SLM) | 1.5–2.1 TB/s | [M] |
| h2d / d2h transfer | ~45 GB/s | [M] |

Use **105.9**, not 136.5, as the denominator when judging efficiency. `dmidecode`
reports two controllers each "Data Width: 16 bits" — that is the per-die LPDDR5
device width, **not** the bus. Do not read 32-bit off it.

### 1.2 Compute

| unit | throughput | source |
|---|---|---|
| vector fp16 | 9.18 TFLOPS | [M] clpeak |
| vector fp32 | 4.76 TFLOPS | [M] |
| vector bf16 | 4.61 TFLOPS | [M] |
| **XMX coopmat fp16** | **25.51 TFLOPS** | [M] clpeak @ 8x8x16 |
| XMX coopmat bf16 | 26.57 TFLOPS | [M] |
| XMX int8 (8x8x32 s8→s32) | ~51 TOPS | [D] 2× fp16 rate, not directly benched |

XMX is ~2.8× the vector path.

### 1.3 Latency

| quantity | value |
|---|---|
| kernel dispatch | **39.8 µs** [M] |
| dispatch round-trip | 52.6 µs [M] |

This is the single most under-appreciated number on this box. See §3.

### 1.4 Power and thermals — nothing is throttling

Measured under sustained LLM load (GPU busy, 80 W PL1 profile):

```
package draw   32.7 W   of 80 W PL1        [M]
  core          8.6 W
  uncore/GPU   14.0 W
GPU frequency  2350 MHz = max = rp0_freq   [M]
package temp   61–65 °C  (limit 105 °C)    [M]
core temps     54–58 °C
```

**The hardware is completely unconstrained.** Max clock, 40 °C of thermal
headroom, drawing 41% of available power. Stalled EUs waiting on memory draw
little power — this *is* the signature of a bandwidth-bound workload.

Consequences:
- Raising PL1 (50 W → 80 W tier) **cannot help**; we are nowhere near the limit.
- Unlocking clocks cannot help; already at rp0.
- **[R]** "the performance power profile is worth +5–7%" — retracted, that was
  noise (§6.1). At a 33 W draw it could not have been real.
- **[R]** "long generations slow down because of thermal throttling" — retracted
  twice-asserted and wrong; see §5.4 for the real cause.

`energy_uj` needs root. `powerprofilesctl` is **not installed**; use `busctl` against
`org.freedesktop.UPower.PowerProfiles` (served by `yoga-powerd`).

---

## 2. XMX: what the matmul primitive actually is, and who can reach it

The hardware unit is Intel **XMX / DPAS** (Dot Product Accumulate Systolic),
8-wide × 8-deep systolic, K set by datatype.

**Exactly four cooperative-matrix shapes are exposed** — probed directly with
`vkGetPhysicalDeviceCooperativeMatrixPropertiesKHR` ([M]; `vulkaninfo` does not
print the shape table and `GGML_VULKAN_DEBUG` is compile-time only, so this needed
a ~50-line C probe):

```
8x8x16   f16  x f16  -> f32
8x8x16   bf16 x bf16 -> f32
8x8x32   s8          -> s32
8x8x32   u8          -> u32
```

**There is no 16x16x16.**

### 2.1 llama.cpp Vulkan can never use XMX here

- It gates its fast coopmat matmul on `coopmat_support_16x16x16_f32acc`
  (`ggml-vulkan.cpp:3646`). With only 8x8x16 it drives 8x8 tiles through 32-wide
  subgroups ⇒ enabling coopmat makes **prefill 2.35× WORSE** (Ornith-9B Q4:
  pp512 218.8 → 92.9) [M]. Decode is unaffected (14.78 → 14.72) because batch-1
  decode is GEMV and never takes a coopmat path.
- `VK_NV_cooperative_matrix2` is **advertised but entirely stubbed** in anv:
  `flexibleDimensionsMaxDimension = 0`, zero flexible-dimension entries [M].
  llama.cpp needs ≥512 plus fp16/fp32 granularities at 128/256 wg-invocations.
  Dead end.
- `GGML_VK_ENABLE_INTEL_COOPMAT=1` on the MoE path is 1.9× slower [M].

⇒ **Do not try to make llama.cpp Vulkan use XMX. Route prefill through OpenVINO.**

### 2.2 OpenVINO does reach DPAS

OV does not go through Vulkan at all — oneDNN + IGC over OpenCL/Level Zero.
It achieves ~63% of XMX fp16 peak [D] (876 t/s prefill on Qwen3.5-9B int4 =
16.1 TFLOP/s against the 25.51 ceiling). That is the entire reason prefill lives
on OV: Vulkan gets 0% of XMX, OV gets 63%.

### 2.3 The int8 DPAS is completely idle — and unreachable from user space

`8x8x32 s8→s32` is ~51 TOPS, ~2× the fp16 rate. OV **dequantises int4 weights to
f16** and uses the fp16 path, so this capability is unused.

Reaching it needs int8 **activations** (W4A8 or W8A8), not just int8 weights:
- **W8A8 is the wrong target**: int8 weights are 1 byte where our experts are
  0.5, so expert bytes per forward would *double*. Decode is bandwidth-bound ⇒
  trades decode for prefill. [D]
- **W4A8 is the right target** (read 4-bit, expand to int8 in registers, DPAS as
  s8→s32) — get 2× compute without 2× bandwidth. This is what QServe/QoQ target.
- OV *does* expose it: `DYNAMIC_QUANTIZATION_GROUP_SIZE` (default 0 = off).

**Measured on the 35B (ctx 8192, no draft) [M]:**

| dqgs | decode t/s | prefill t/s |
|---|---|---|
| unset (default) | 23.32 | **746.8** |
| 0 (explicit off) | 22.19 | 740.4 |
| 32 | 22.72 | 707.2 |
| 64 | 23.31 | 698.9 |
| 128 | 22.34 | 726.8 |

**No configuration beats the default.** 32/64 are real regressions (−5…−6%).
Note `-1` vs `0` differ by 0.9% though they are the same thing — that is the
noise floor.

**Probable reason [D, not proven]:** our prefill runs through the plugin's
*custom fused* MoE kernels — `moe_3gemm_swiglu_opt` and `moe_router_fused_opt`,
verified present in the runtime graph — and a generic activation-quantisation
hint cannot change what a hardcoded fused kernel does internally. Ironically the
kernels that give us a 2× prefill advantage over llama.cpp are what block the
int8 path. Getting there needs Intel to ship int8 MoE kernels.

Independent blocker: activation quantisation in LLMs hits emergent outliers,
needing SmoothQuant/AWQ-class mitigation, and no recipe exists for `qwen3_5_moe`,
let alone one that survives a hybrid GatedDeltaNet's recurrent state.

---

## 3. Roofline: why decode is hopeless and prefill is not

### 3.1 Arithmetic intensity at batch 1

Each weight is read once and used in one multiply-add ⇒ 2 FLOP per weight. At
int4 (0.5 byte) that is **~4 FLOP/byte**.

Ridge point (where compute and bandwidth balance):
- vector fp16: 9.18 TFLOPS / 105.9 GB/s = **87 FLOP/byte**
- XMX fp16: 25.51 / 105.9 = **241 FLOP/byte**

**We are 20–60× to the wrong side of the ridge.** This is inherent to
autoregressive batch-1 decoding and no architecture changes it. The only things
that raise decode AI are batching (irrelevant, single user) and **speculation**.

### 3.2 Bytes per token

| model | composition | bytes/token | ceiling = 105.9/bytes |
|---|---|---|---|
| 35B-A3B int4 | 2.43 GB dense (i8) + 1.02 GB experts (u4, 8/256 active) | **2.94 GB** | **36.0 t/s** [D] |
| 9B int4 | 6.97 G params u4 + 2.04 G u8, all active (dense) | **5.52 GB** | **19.2 t/s** [D] |

The 9B reads **1.9× more per token** than the 35B despite being 4× smaller, because
it is dense and the 35B is sparse (8 of 256 experts).

### 3.3 Where each model sits

| | no-draft decode [M] | its own ceiling [D] | efficiency |
|---|---|---|---|
| 9B | 18.6–19.7 t/s | 19.2 | **~97% — bandwidth-bound, maxed** |
| 35B | 22.97 t/s | 36.0 | 64% — **dispatch-bound** |

### 3.4 The dispatch term

~540 kernel launches per target forward × 39.8 µs = **21.5 ms** [D from measured
residual]. Compare bandwidth time at nat=6: 45.9 ms. So dispatch is ~32% of a
forward and is *not* recoverable by bandwidth tricks.

Residual per-token latency R (measured − bytes/105.9) tracks **kernel count**, not
bytes [M]: 9B dense 17.7 ms, 35B-A3B 21.6, Gemma-4-26B-A4B 20.4, LFM2-24B-A2B 10.5.
LFM2 is half the dispatch cost of the others ⇒ **shallow-and-wide beats
deep-and-narrow on this chip**, independent of FLOPs.

### 3.5 Full gap analysis at nat=6 (35B)

```
bandwidth   45.9 ms   (2.43 GB dense + 2.43 GB expert union of ~38 experts)
dispatch    21.5 ms   (~540 kernels x 39.8 us)
draft       ~10 ms    (737 MB DFlash draft)
            -------
            ~77 ms -> 3.59 accepted tokens -> ~46 t/s predicted
measured                                      ~37 t/s
pure-bandwidth ceiling                          78 t/s
```

The no-draft case validates well (predicted 20.8 vs measured 22.97, 7%). The
speculative case **overpredicts by ~25%** even after adding the draft — residual
unexplained. Do not treat the 46 t/s as reachable headroom without finding it.

### 3.6 Below ~2 bits this machine becomes dispatch-bound

Bandwidth and dispatch terms are currently comparable (45.9 vs 21.5 ms). Go
ternary (1.58-bit, ~2.5× fewer bytes) and bandwidth drops to ~9 ms while dispatch
stays 21.5 ⇒ **capped near 46 t/s by kernel launches alone** [D]. So *fewer
kernels* matters more than *fewer bits* past that point. This is the strongest
argument for a shallow-wide architecture over an aggressively-quantised one.

---

## 4. Prefill

### 4.1 Prefill is governed by CHUNK COUNT, not context length

`max_num_batched_tokens` (mnbt) sets the prefill chunk; ⌈ctx/mnbt⌉ chunks each
cost a pass.

**mnbt sweep at ctx 32768, 35B [M]:**

| mnbt | chunks | prefill t/s | footprint |
|---|---|---|---|
| 2,048 | 16 | 453.2 | 22.60 GB |
| 4,096 | 8 | 477.7 | 22.81 GB |
| 8,448 (shipped) | 4 | 499.4 | 23.76 GB |
| 16,384 | 2 | 532.5 | 24.50 GB |
| 32,768 | **1** | **604.1** | 26.88 GB |

**+21% prefill for +3.1 GB.** This explains the apparent "cliff": 8k with
mnbt 8448 is *one* chunk (737 t/s) and 32k is *four* (499). Only ~18% of that gap
is genuine context decay — 32k in a single chunk recovers to 604.

⇒ **Prefill speed trades directly against memory, and memory is what long context
needs. You cannot have both.**

### 4.2 PERFORMANCE_HINT / NUM_STREAMS

`PERFORMANCE_HINT=LATENCY` + `NUM_STREAMS=1` [M]:
- **+58% prefill at 32k** (313 → 494)
- ~+4% at 8k
- +5% at 64k

It matters wherever prefill is chunked. The default THROUGHPUT hint sizes
per-stream buffers for concurrency this box cannot afford. `bench_35b.py`
historically omitted both while `ornith35_ov.py` set them, so **every bench number
taken through that harness understated long-prompt prefill.**

### 4.3 Measured prefill curve (35B, LATENCY, mnbt 8448 unless noted)

| ctx | prefill t/s | footprint |
|---|---|---|
| 2,048 | 695–715 | 21.19 GB |
| 6,144 | 725 | — |
| 8,192 | 737.0 | 20.59 GB |
| 32,768 | 499.4 (604.1 single-chunk) | 23.76 |
| 65,536 | 355.8 | 24.78 |
| 100,000 | 274.1 | 24.69 |
| 131,072 | 217.5 (mnbt 2048, cache 3) | 22.69 |

9B prefill: **~1370 t/s** at 2k–8k [M] — roughly **2× the 35B**, and flat with
context. The 9B's one clear win.

### 4.4 [R] The "1808 t/s @6k" figure is not real

Retracted. Re-tested exhaustively across draft on/off, `dynamic_split_fuse`,
`PERFORMANCE_HINT`, `NUM_STREAMS`, mnbt 2048→32768, ctx, and rep warming: the
ceiling is **~740**. Almost certainly a bad `num_input_tokens / TTFT` ratio in the
original note. It was the headline number in two places for days.

### 4.5 MoE prefill scaling is KERNEL-specific, not architectural

Qwen3.5-35B-A3B reaches 23.66 TFLOP/s at 436 rows/expert; Gemma-4-26B-A4B at the
**same** 436 rows/expert manages only 4.10 TFLOP/s — 5.8× worse [M]. It is a
property of `qwen3_5_moe`'s fused `moe_3gemm_swiglu_opt` path. **Do not use the
rows-per-expert law to predict other architectures.**

Also: rows/expert = tokens·k/E, so *higher* expert count gives *fewer* rows each —
"more experts ⇒ more rows/expert" is inverted.

---

## 5. Decode

### 5.1 Speculation is the whole game

| 35B config | decode t/s [M] |
|---|---|
| no draft | 22.97 |
| DFlash nat=4 | 34.77 |
| DFlash nat=6 | 36.91 |
| DFlash nat=8 | 37.04 |

**1.6×.** All nat ≥4 differences are inside the noise floor (§6.1) — treat 4/6/8
as a tie and do not tune it. 9B: 19.69 → 35.6–35.8 (also 1.7×, also a tie across
nat 5/7).

### 5.2 Acceptance is where the remaining ~2× lives

```
decode t/s = A / (target_time + draft_time),   A = 1 + n·acceptance
```

At nat=6, 43.1% acceptance ⇒ A=3.59 ⇒ 36 t/s. Holding target_time fixed (the
expert union depends on *b*, not on acceptance):

| acceptance | A at n=6 | predicted decode [D] |
|---|---|---|
| 43% (today) | 3.59 | 36 t/s |
| 60% | 4.60 | 60 t/s |
| 80% (what a *distilled* head achieves) | 5.80 | 75 t/s |

Our draft is **base-Qwen3.6, not Ornith** — that is why it sits at 43%. No
Ornith-35B DFlash draft exists anywhere (surveyed HF: only 397B variants, a 19.4 GB
AMD bundle tagged `cross-model-draft`, and DSPARK drafts with incompatible tap
counts). It has to be distilled.

### 5.3 The MoE expert-union penalty is real and measured

Verifying b tokens touches the **union** of experts across them:
E[distinct] = 256·(1−(1−8/256)^b) under independence — b=1→8, b=5→37.6, b=9→63.6.

But routing is **not** independent: measured expert overlap between consecutive
tokens is **38.1% vs 6.25% uniform**, and temporal correlation cuts unique experts
by **20–31%** vs the independence baseline (Cohere, k=8/N=128). So the union grows
more slowly than the naive formula — my "routing is close to independent" claim
was **[R]** wrong.

Observable consequence [M]: acceptance decays 72.5% → 32.4% as nat goes 2 → 8, and
throughput *peaks* then *declines*. Deeper drafting stops paying because the union
grows faster than A. **High expert count is a liability for prefill and an asset
for speculative decode** (at b=9 a 256-expert model touches 24.8% of its expert
block; a 128-expert model would touch 44.4%).

Literature targeting exactly this: [EcoSpec](https://arxiv.org/abs/2607.12696)
(cost-aware draft selection, up to 1.62×), [MoE-Spec](https://arxiv.org/abs/2602.16052)
(training-free verification-time expert budgeting, 10–30% over EAGLE-3),
[Adaptive Verification](https://arxiv.org/html/2605.00342),
[DraftExpert](https://arxiv.org/html/2607.24434).
**Caveat specific to this box:** those papers optimise expert *bytes* because
datacentre GPUs have cheap dispatch. We are ~32% dispatch-bound, so byte budgeting
should help *less* here than their headline numbers.

### 5.4 Decode and acceptance both DECAY with generation length

This is the answer to "why does a long run average ~6 t/s when the benchmark says 34":

| generated tokens | 9B decode [M] | acceptance [M] |
|---|---|---|
| 128 | 35.6 t/s | 44.6% |
| 4,000 | **18.2 t/s** | **26.5%** |
| ~15–20k | ~6 t/s | — |

Two compounding causes: growing-KV attention cost, and drafter degradation (visible
in the logs as `draft=[279,279,279,279,279]`, the same token five times, accepted 0).

⇒ **Every t/s number in this document is a 128-token burst and overstates real
long-form work by 2–6×.** Not thermal (§1.4).

### 5.5 MTP vs DFlash: DFlash wins

llama.cpp 35B, same pinned power tier, 3 reps [M]:

| config | generation t/s | within-config sd |
|---|---|---|
| **DFlash n=2** | **33.45** | 0.07 |
| MTP n=2 | 32.00 | 0.40 |
| MTP n=3 | 30.6 | — |

4.5% gap against a 1.2% sd ⇒ real. And both lose to OV+DFlash's 37.04.

**[R]** An earlier reading had MTP marginally *ahead* (33.1 vs 32.8) — that was
taken across drifting power tiers.

Two structural reasons MTP cannot win here:
1. **MTP is autoregressive** — n draft forwards for n tokens — while DFlash proposes
   the whole block in **one** forward. On a box where 21.5 ms of every 67 ms forward
   is dispatch, that dominates head quality.
2. **MTP's head carries a duplicate `lm_head`** — 0.509 GB at u8 for the 35B, read
   *per drafted token*. At n=6 that is ~29 ms vs DFlash's ~10 ms for the whole block.
   (The 9B mitigation was requantising the head's `token_embd`+`output` to Q2_K,
   2.26 → 0.87 GiB, which took decode 18.8 → 22.4 t/s [M]. Unapplied to the 35B.)

### 5.6 DFlash constrains decoding to greedy

`dflash_strategy.cpp:415` asserts **"DFlash CB/PA currently supports greedy decoding
only."** So `do_sample=true` is unavailable while speculating — the model's own
recommended `temperature/top_k/top_p` (35B ships `do_sample:true, temp 1.0,
top_k 20, top_p 0.95`) cannot be used on the fast path.

`repetition_penalty` **is** usable: it is a logit processor and does not change
`is_greedy_decoding()`. This matters enormously — see §7.

### 5.7 Quant type, not shaders, was the biggest single decode win

`Ornith-1.0-35B-unsloth-UD-IQ3_XXS` stores experts as IQ2_S/IQ3_S/IQ4_XS, **none of
which are on llama.cpp's integer-dot allowlist** for the MoE mat-vec path
(`ggml_vk_get_dequantize_mul_mat_vec_id`, `ggml-vulkan.cpp:7658`), so every expert
matmul silently fell back to a slow f32-dequant shader on a device advertising
`int dot: 1`. Same-shape proof: `iq2_s` 15.6 GB/s vs `q4_K` **58.0 GB/s** [M].

Requantising to Q4_K_M: pp512 134.9 → 247.4, tg128 17.66 → 22.71, tg+DFlash
21.2 → 32.8 [M].

Also `[[dont_unroll]]` in `mul_mm.comp` is a **win, not a tax**: unrolled is 18–24%
slower, 2,456 instr / 0 spills rolled vs 4,797 / 73 spills unrolled [M].

---

## 6. Measurement methodology — read this before trusting any number

### 6.1 The noise floor is 5.4% CV, so differences under ~11% are meaningless

Five **identical** runs of one config (35B nat=8, ctx 2048, reps=3, drain between,
pinned power) [M]:

```
33.36  37.74  34.45  33.05  34.23     mean 34.57  sd 1.87  CV 5.4%
range 13.6%
within a single process, reps span up to 11.5%
prefill threw one 363 t/s outlier against ~690 elsewhere (−47%)
=> at n=1, a difference must exceed ~11% to mean anything
```

Cause is **not** thermal or power (§1.4) and remains unexplained. llama.cpp is
dramatically more reproducible (sd 0.4 on 3 runs) than OV.

This retroactively kills: the nat 6-vs-8 ranking, the whole 35B and 9B nat sweeps,
the dqgs sweep, the power-profile effect, and the 9B-vs-35B decode comparison
(37.04 vs 35.77 = 3.5%). **The only defensible 35B decode figure is ~34.6 ± 1.9 t/s.**

What survives: draft vs no-draft (+50%), 9B vs 35B *prefill* (2×), mnbt at 32k
(+33%), LATENCY at 32k (+58%), DFlash vs MTP on llama.cpp (4.5% against 1.2% sd).

### 6.2 Pin the power profile before any timing bench

`yoga-powerd` serves `org.freedesktop.UPower.PowerProfiles` and switches tiers
underneath you. It started mid-session at `base=balanced` and only went to
`performance` 2h15m later, silently invalidating an afternoon of numbers and
flipping two conclusions. Pin via `busctl` set `ActiveProfile=performance`, then
**log** `platform_profile`, `energy_performance_preference`, RAPL
`constraint_0_power_limit_uw`, and the D-Bus profile before **every** run.

**Acceptance is power-independent** — greedy/temp-0 is deterministic, so
55.1282/43.0857/32.3939% reproduced to the digit across tiers. Only t/s needs
re-measuring after a profile change. Useful for distinguishing a real regression
from a power artifact.

### 6.3 The warm-up-line trap

`bench_35b.py` emits DFlash acceptance lines for its **warm-up** generations too.
Only trust lines with `generated_len` equal to `--max-new`; the `generated_len=8`
lines are 6-token samples. Reading them made a healthy GPU draft look
catastrophically broken (a fake "50% → 15% → 13.7% collapse") and made nat=2 look
optimal. Both conclusions were published before being caught.

### 6.4 Other traps

- **Reuse of a pipeline across growing prompt lengths** trips a GPU realloc bug and
  fabricates a ceiling. `bench_35b.py` builds the pipeline *before* its `--ctx`
  loop, so a multi-ctx invocation is invalid — one ctx per process.
- **`ContinuousBatchingPipeline` must take FOUR positional args.** The 5-arg
  overload's 5th param is `tokenizer_properties`, so passing `{}` then
  `{"draft_model": ...}` **silently discards the draft** and you measure plain CB
  while believing speculation is on. This invalidated an entire round of results.
- **`-no-cnv` does not suppress conversation mode** in this llama.cpp build; runs
  hang on an interactive `>`. Use `-st` plus `</dev/null`.
- **`llama-bench` has no draft support**; only `llama-cli` can measure speculation.
- **`MemoryMax=` gives no OOM protection.** GPU/TTM pages are invisible to both RSS
  and cgroup accounting: a 131k-context attempt produced a *global* OOM with
  `anon-rss: 5056kB`, killing unrelated user processes. Bound memory by choosing
  conservative mnbt/cache_gib, never by trusting the cgroup limit.
- **Warm-up is mandatory** for prefill: cold ~483, warm ~700.
- **`TMPDIR` must not be `/tmp`** (16 GiB tmpfs); ENOSPC surfaces only as
  `RuntimeError: basic_ios::clear: iostream error`.

---

## 7. Decoding config: greedy causes rumination loops

Both OV runners hardcoded `cfg.do_sample = False` — a *benchmarking* requirement
(greedy is needed for deterministic KLD and bit-identical acceptance) that leaked
into normal use. Greedy + a thinking model ruminates.

**Measured on the 9B, same 4000-token budget [M]:**

| | distinct-line ratio | reached code? |
|---|---|---|
| greedy, no penalty | 0.62 (13× the same line) | **no** — still deliberating |
| greedy + `rep_penalty 1.1` | **0.89** | **yes** |

With an 18k budget and the penalty, the 9B **terminated naturally at 7,574 tokens**
instead of ruminating to the cap. Cost: acceptance 26.5% → 22.6%, decode flat
(18.19 → 18.28) — **essentially free**.

Prior symptom: a 60-minute run produced **0 characters** because it never emitted
EOS and the script only writes at the end.

llama.cpp ships `--repeat-penalty 1.00` (**disabled**) in this build, so its
production modes had no protection either. All four paths now set 1.1.

---

## 8. Context

KV is cheap because the model is **hybrid**: only **10 of 40 layers** keep a KV
cache (layers 3,7,11,…,39); the other 30 are GatedDeltaNet with fixed-size
recurrent state.

```
KV = (512 + 512) x 2 B x 10 layers = 20,480 B = 20.0 KiB/token   [D]
measured: 15 KiB/token (8k vs 32k at fixed pool)                 [M]
```

**Footprint is set by mnbt and cache_gib, NOT by context length** — 131k costs
*less* (22.69 GB) than 100k did (24.69 GB) purely because mnbt was 2048 vs 8448.
At fixed mnbt, 64k → 100k moved footprint by 0.09 GB.

**Full 131,072 context is usable**: 217.5 t/s prefill, 21.3 t/s decode, 22.69 GB,
coherent output [M]. It **OOMs** at `cache_gib=4 --mnbt 8448` — that was a config
error, not a capacity limit.

---

## 9. Quality: where the speed numbers stop mattering

KLD vs a Q8_0 reference, 16,384 held-out agentic positions, GPU [M]:

| 35B config | mean KLD | median | same top-1 |
|---|---|---|---|
| GGUF transplant (lossy-on-lossy) | 0.3458 | 0.01369 | 87.30% |
| **bf16 requant** | **0.1596** | **0.00416** | **92.60%** |

2.17× better at **zero speed cost** (byte-identical layout).

9B int4 vs a **bf16** reference: mean KLD **0.7220**, top-1 **81.98%** — and it is
already at the u4/g128 floor (plain rel-L2 0.1037 ≈ step/√12 ≈ 0.104σ), so this is
the price of 4-bit on a 9B, not a defect. Both improvement routes were measured and
rejected: the MSE shrink grid buys 5% weight error; `ratio 0.8` buys 0.70 points of
top-1 for +13% bytes/token (≈7 t/s).

**Why the 35B is good and also slower**: ~71% of its *active* weight is int8 (the
dense attention/shared-expert path) and only the sparse experts are int4. The 9B is
dense so ~77% of its active weight is int4. Upstream's `Qwen3.6-35B-A3B-int4-ov`
carries 1.27 GB of non-expert vs our 2.43 GB — faster and worse. **We bought
quality with bandwidth.**

### 9.1 KLD badly understates real-task failure

A single coding task (physics simulation, ~10 interacting requirements):

| | 35B | 9B |
|---|---|---|
| code emitted | 42,147 chars | 20,453 chars |
| syntax | clean first try | **13 dropped tokens** (9× missing `.radius`, 4× missing `*`) |
| parses | ✅ | ❌ `p.x = TANK_RIGHT_X - p.;` |
| renders | tank, 3 fish, plants, rocks, floating toy, gauges | scattered dots + a misplaced line |
| defect | burst trigger unreachable (`else if (state==='stable')` after `if (state==='stable')`) | most of the brief unimplemented |

**Nine dropped tokens out of ~5,000 barely move an average KLD but make the artifact
worthless.** The dropped tokens are high-probability/low-information (`radius`, `*`)
— exactly what a model at 82% top-1 drops. Distributional metrics on next-token
prediction cannot see this. **Always run a real task eval before believing a
quality ranking.**

---

## 10. What is actually reachable

Ranked by confidence, for the 35B on OV:

| lever | expected | status |
|---|---|---|
| `--mnbt` for long prompts | +21% prefill at 32k | **measured, exposed via CLI** |
| `PERFORMANCE_HINT=LATENCY` + `NUM_STREAMS=1` | +58% prefill at 32k | **measured, shipped** |
| bf16 requant | 2.17× quality, free | **measured, shipped** |
| `rep_penalty 1.1` | makes long generations terminate at all | **measured, shipped** |
| Ornith-distilled DFlash draft | 43% → ~80% acceptance ⇒ ~36 → ~75 t/s | **not built; nothing to download** |
| MoE-Spec expert budgeting | targets 2.43 GB of 4.86 GB per forward | training-free; expect < paper's 10–30% here (dispatch-bound) |
| DSpark (Markov + confidence heads) | unquantified | drafts exist in BF16 with matching tap contract; needs head support |
| int4 dense path | −24% bytes/forward | **would destroy the quality win** |
| W4A8 / int8 XMX | ~2× prefill compute ceiling | **measured no gain**; blocked by fused MoE kernels |
| raise PL1 / unlock clocks | **zero** | already at 33 W of 80 W and max clock |

### Theoretical spec sheet for this box

Natively low-bit (QAT int4 or ternary, to escape the quantisation floor), **int8
activations** to reach the idle s8→s32 DPAS, sparse MoE with many small experts and
low top-k (minimise bytes/token *and* the speculative union), **shallow and wide**
to cut the 540-kernel dispatch bill, hybrid linear attention to keep KV near-zero,
and a native MTP/DFlash head distilled against the *actual* target for high
acceptance.

Qwen3.x-35B-A3B already provides the sparse-MoE, hybrid-attention and speculation
pieces — which is why it keeps winning here. Nothing available provides int8
activations at competitive quality. **Qwen3.x-35B-A3B is the optimal available model
for this box; do not switch.** Dense is a trap at this size: Qwen3.6-27B dense reads
13.93 GB/token ⇒ 6.5 t/s decode. Qwen3.5-122B-A10B is arithmetically excluded
(experts alone are 60.3 GB at int4).

---

## 11. Qwen3.6-35B-A3B and DSpark (2026-08-01)

### 11.1 Qwen3.6 wins on BOTH axes — the int4-dense trade did not cost quality

Downloaded `OpenVINO/Qwen3.6-35B-A3B-int4-ov` (19.65 GB), then two pure graph
surgeries, no weight touched: **fuse the separate `text_embeddings` Gather into the
LM** (`fuse_embed.py`; VLM exports take `inputs_embeds`, and OV GenAI refuses
DFlash on a model with an embedder) and **convert M-RoPE `position_ids [4,B,S]` to
plain `[B,S]`** (`derope_mrope.py`). Result: `Qwen36-35B-int4-ov-cb`.

Precision layout [M]: **97.1% u4 / 1.4% 8-bit**, vs arm B's 92.4% u4 / 6.9% 8-bit.
Qwen3.6 holds its always-active dense path at int4 where arm B holds it at int8.

| | decode (no draft) | prefill @2k | footprint |
|---|---|---|---|
| Ornith arm B | 22.97 t/s | ~700 | 21.19 GB |
| **Qwen3.6** | **31.3 t/s (+36%)** | 610–666 | 20.0–22.7 GB |

**+36% decode clears the 5.4% noise floor comfortably** (predicted +24% from
bytes/token 2.38 vs 2.94 GB). Prefill ~5% lower = inside noise, no claim.

**And it also wins the coding eval** — I predicted it would trade quality for speed
and it did not. Model generation beat quantisation precision.

⚠ A `longjmp causes uninitialized stack frame` crash at `cache_gib=2 --mnbt 8448`
was **transient contention with another process's residual GPU allocations**, not
the model. Diagnostic order that settled it: CPU compile OK → GPU *plain*
compile OK (argmax identical to CPU) → only the CB path failed ⇒ allocation, not
graph validity. The error text is actively misleading.

### 11.2 The three-way coding eval — severity ordering KLD cannot produce

One prompt (aquarium burst sim, ~10 interacting physics requirements):

| | 9B | Ornith 35B | **Qwen3.6 35B** |
|---|---|---|---|
| code | 20,453 ch | 42,147 ch | 32,395 ch |
| parses | ❌ **13 dropped tokens** (9× missing `.radius`, 4× missing `*`) | ✅ | ✅ |
| renders | scattered dots | tank/fish/plants/rocks/toy/gauges but **water unfilled, fish above waterline** | **filled water, wavy surface, 8 fish, floating duck, % scale, real buttons** |
| burst | n/a | ❌ **trigger unreachable** — `else if (state==='stable')` after `if (state==='stable')` | ✅ **drains, waterline falls continuously, duck tracks surface, puddle forms** |
| defect | most of brief unimplemented | no simulation possible | `const crackStartY` reassigned ⇒ throws mid-`triggerBurst`, so no glass fragments AND crack drag cannot affect the jet |

**None of the three produced correct code.** All had bugs one run would catch. The
useful output is the ordering: *doesn't parse* < *runs but the feature is
unreachable* < *runs, feature works, secondary effects missing*.

### 11.3 DSpark is NOT a drop-in DFlash — its extra heads are load-bearing

DSpark = DFlash + a **Markov logit-bias head** (each draft position conditions on
previously-sampled tokens within the block) + a **confidence head** (per-position
acceptance prediction). Its config class subclasses `DFlashSpeculatorConfig` and
"inherits all DFlash fields unchanged", and its DFlash-subset tensors are named
identically — so it *exports and runs* through our DFlash path with two patches
(accept `Qwen3DSparkModel`; drop `markov_head.*`, `confidence_head.*`,
`embed_tokens.*`, `lm_head.*` — the runtime clones the last two from the target).

**But amputated, it is far worse than no draft at all [M]:**

| Qwen3.6 target, nat=6 | acceptance | decode |
|---|---|---|
| no draft | – | **31.3 t/s** |
| DSpark draft, GPU | 3.29% | 14.56 t/s |
| DSpark draft, CPU | **3.29% (identical to the digit)** | 13.51 t/s |

CPU == GPU acceptance **rules out** the GPU SDPA-fusion bug that once collapsed the
9B draft to 0.32%. The cause is the amputation: the body was trained expecting the
Markov bias and its *own* `lm_head`, and gets neither.

⇒ Getting DSpark's actual advantage needs the Markov head built into the exported
graph **and** a matching strategy in the genai fork. Not an export problem.

Also useful: `annotate_dflash_taps.py` is required for a freshly-composed target —
the genai fallback pattern matcher fails with
`Check 'outputs.size() == target_layer_ids.size()' failed`
(`dflash_model_transforms.cpp:233`) even though the 81 `aten::add/Add_1` residual
nodes are present and identically named to a working target. Note its `--suffix`
is the NODE-name suffix (default `Add_1`), not an output-dir suffix, and it edits
the target **in place** (backing up the `.xml`).

### 11.4 Speculation pays LESS on a faster baseline — and that flips the recommendation

Provenance-matched for the first time (Qwen3.6 target + `z-lab` Qwen3.6 DFlash draft,
on the tap-annotated target) [M]:

| config | no draft | + DFlash | speculation gain |
|---|---|---|---|
| Ornith arm B | 22.97 | **37.04** (nat 8) | **+61%** |
| Qwen3.6 | **31.3** | 34.1 (nat 6, acc 32%) | **+9%** |
| Qwen3.6 | – | 32.5 (nat 4, acc 42%) | +4% |

A draft's cost is roughly fixed (~10 ms: 737 MB + its dispatch), so the FASTER the
target forward, the larger that tax is as a fraction — speculation has less room to
pay. Qwen3.6 decodes 36% faster unspeculated and therefore gains only +9% from a
draft, versus Ornith's +61%.

**37.04 vs 34.1 is 8.6%, inside 2σ of the 5.4% noise floor ⇒ peak decode is a TIE.**
Only Qwen3.6's no-draft advantage (+36%) is outside noise.

⇒ **Plain Qwen3.6, no draft, is the better production config** even though peak
decode ties:
- 31.3 t/s with no drafter to export, tune, or keep provenance-matched
- **20.03 GB vs ~24 GB** (the draft costs 3.6 GB) — buys back context headroom
- wins the coding eval (§11.2)
- **escapes the DFlash greedy-only assert** (§5.6), so the model's own recommended
  `temperature/top_k/top_p` become usable and the greedy rumination-loop exposure
  (§7) disappears rather than needing `rep_penalty` to paper over it.

This is the one case in this document where the simpler configuration wins on every
axis that matters, and it only became visible because the no-draft baseline was
measured rather than assumed.

### 11.5 [R] "MTP loops under greedy too" — RETRACTED, rumination is model-capability

I inferred from config (same family, same greedy, no penalty) that llama.cpp MTP
would ruminate like the 9B did. **Measured at 1500 tokens and it does not** [M]:

| 35B + MTP, llama.cpp, greedy | distinct-line ratio | terminated cleanly |
|---|---|---|
| no repetition penalty | **1.00** | yes |
| + `--repeat-penalty 1.1` | 0.98 | yes |

⇒ **Rumination is a WEAK-MODEL failure, not purely a decoding-config one.** The 9B
(mean KLD 0.72, top-1 82%) ruminated 4000 tokens under greedy and never emitted
EOS; the 35B under identical greedy decoding is clean. `rep_penalty` is a
mitigation for weak models — cheap insurance worth keeping as a default, but the
"greedy causes loops" framing of §7 is too strong as stated: greedy causes loops
*in models weak enough to have nothing better to do*.

Caveat on the test: the prompt was abbreviated and the model drifted to writing
image-generation prompts rather than code, so it is not a like-for-like content
comparison — but the repetition measurement stands regardless of content.
Generation was 28.1-28.3 t/s vs 33.3 at 160 tokens, consistent with §5.4 decay.

### 11.6 [R] "DSpark's extra heads are load-bearing" — RETRACTED. It was a QUERY-LAYOUT bug.

I attributed DSpark's 3.29% acceptance to dropping its Markov/confidence heads and
its own `lm_head`. **Wrong on both counts.**

**The `lm_head` was never the issue.** DSpark's shipped `lm_head` and the target's are
the SAME weights: sign-invariant row cosine **0.9999 mean / 1.0000 median / 0.9998 at
p1, 100% of rows > 0.99** [M]. (The raw comparison first showed mean cosine 0.2586
with 37% of rows at exactly −1.0 — that was MY bug: comparing raw i8 codes without
applying the per-row scales, which in this scheme are `signed_extreme / -128` and
therefore NEGATIVE for 37% of rows. The bimodal ±1 split was the tell.) So borrowing
the target's head, as the DFlash export does, is safe — `gate_phaseB_drafts.py`'s
`check_embeddings_match_target` exists to answer exactly this.

**The actual cause was the seed position.** DFlash emits `1 + N` query slots and
DROPS position 0; DSpark emits `N` slots where **the anchor is itself a prediction and
no output position may be dropped**. Our export defaulted to dropping it, shifting
every proposal by one. One flag — `--no-drop-seed-position`:

| Qwen3.6 target, nat=6 | acceptance | decode |
|---|---|---|
| no draft | – | **31.3** |
| DSpark, seed DROPPED (first export) | 3.29% | 14.56 |
| **DSpark, seed KEPT** | **24.49%** | **30.78** |
| DFlash (z-lab, matched provenance) | 32% | 34.1 |

**7.4× acceptance from one flag.** `export_dflash_ov.py`'s own docstring predicted it:
*"If validation shows the drafted tokens are shifted by one, this flag is the first
thing to flip."* Two signals said off-by-one before I theorised — `proposed[1] ==
truth[1]` while slot 0 disagreed, and an alignment sweep peaking sharply at +1 (45.8%
vs 3-8% elsewhere) — and I read them as a quirk of my probe rather than of the export.

### 11.7 What the Markov head is actually worth (PyTorch probe, no C++)

`probe_markov.py` taps the annotated target for its 8 DFlash hidden states while
greedy-decoding, then runs `dspark_torch.DSparkDraftForOV` on those taps with
`with_markov` on/off, sampling the block left-to-right so token k sees the bias of the
token sampled at k−1. Scored against the target's own greedy continuation [M]:

| | per-slot top-1 | prefix acceptance | A |
|---|---|---|---|
| plain | 33.3% | 15.3% | 1.92 |
| **with Markov** | **45.8%** | **25.0%** | **2.50** |

**+38% relative on per-slot agreement.** These are LOWER bounds — the probe runs with
`past_len=0` (no draft KV; all context arrives via `hidden_states`) where the runtime
precomputes proper context KV.

The Markov bias itself is a low-rank transition term, `markov_w2 @ markov_w1[prev]`
with rank 256, and its magnitude is large enough to reorder top-k: std 0.274, absmax
3.58 on a single prev-token [M].

⇒ If that +38% carries into the runtime, DSpark lands ~33-35% acceptance and plausibly
past DFlash's 32%. Implementation is ~80 lines in `dflash_strategy.cpp`'s existing
per-position `sample_candidates` loop (line 154) plus getting `markov_w1/w2` to C++.
**Cost to weigh:** `markov_w2` is [256, 248320] = 63.6 M params, and the bias needs the
whole matrix per position ⇒ **~0.4 GB extra reads per draft round** on a
bandwidth-bound box, i.e. ~+4 ms against a ~10 ms draft.
Incremental rebuild of the genai fork is **10 seconds** [M], so iteration is cheap.

Naming gotcha: the checkpoint stores `markov_head.markov_w{1,2}.weight`; `dspark_torch`
expects `markov_w{1,2}.weight`. Strip the `markov_head.` prefix or nothing loads.

### 11.8 VERDICT: do NOT implement the Markov head — acceptance is not the binding constraint

DSpark nat sweep with the seed-position fix (Qwen3.6 target) [M]:

| nat | acceptance | decode |
|---|---|---|
| 2 | **75.0%** | 31.57 |
| 4 | 37.5% | 28.08 |
| 6 | 24.5% | 30.78 |
| 8 | 33.0% | 23.96 |
| no draft | – | **31.3** |
| DFlash nat=6 | 32% | **34.1** |

**At nat=2, DSpark hits 75% acceptance and decode is still just 31.57 — parity with no
draft.** So on this target, tripling acceptance buys nothing: the draft's fixed cost
against a fast target forward absorbs the entire gain. Acceptance is NOT the limiting
factor.

⇒ **The Markov head is not worth implementing here.** Its measured +38% relative lift
(§11.7) would move a quantity that demonstrably does not control throughput on this
target. ~80 lines of C++, `markov_w2` plumbing and ~0.4 GB/round of extra reads for
~zero.

This is the same lesson as §11.4 from the other direction: **speculation only pays when
the target forward is slow enough for the draft to be cheap by comparison.** On Ornith
(22.97 t/s baseline) DFlash returns +61%; on Qwen3.6 (31.3 t/s baseline) nothing
returns more than noise. Before optimising a drafter, check that the baseline leaves
room for one.

The DSpark work was still worth doing: it produced the seed-position fix (§11.6, 7.4x),
proved the `lm_head` equivalence, and quantified the Markov head — all of which would
matter on a SLOWER target where speculation has headroom.

### 11.9 [R] "The int8 DPAS is completely idle" — RETRACTED. The DENSE path already uses it; the EXPERTS are what's stranded.

Read straight off the compiled GPU runtime graph (`compile_model(...).get_runtime_model()`,
`rt_info['runtimePrecision'].astype(str)` — note OVAny needs `.astype`, `str()` yields
`<OVAny class>`) [M]:

```
371  FullyConnected              i8      <- dense path EXECUTES IN INT8
371  DynamicQuantize             f16     <- one per FC: activations quantised to int8
 40  moe_3gemm_fused_compressed  f16     <- experts dequantised to f16
 40  moe_router_fused            f16
```

Identical counts for Ornith arm B and Qwen3.6. So §2.3 was wrong twice over:
- **The `8x8x32 s8->s32` DPAS is already reached** for every dense matmul, with
  dynamically-quantised int8 activations. We do NOT lack W4A8; it ships enabled.
- This also explains the `DYNAMIC_QUANTIZATION_GROUP_SIZE` null result correctly:
  dynamic activation quantisation is **already on by default** for those 371 layers,
  and the experts are fused kernels that ignore the hint. I had guessed the
  fused-kernel half and missed that the other half was already enabled.

**EXECUTION PRECISION IS SET BY KERNEL AVAILABILITY, NOT BY WEIGHT FORMAT.** The
decisive pair: Qwen3.6's dense weights are `u4` and execute at `i8`, while the *same*
`u4` format in the experts executes at `f16`. The plugin up-converts u4->int8 for
`FullyConnected` but has no int8 path in the MoE primitive — `MOECompressed` /
`moe_3gemm_fused_compressed` is the only compressed-MoE primitive registered, and no
int8 variant appears in the plugin symbol table.

⇒ **Do NOT quantise the experts to int8 hoping to reach the DPAS.** It would double
active expert bytes (0.54 -> 1.08 GB), push bytes/token 2.95 -> 3.49 GB and decode
ceiling 35.9 -> 30.3 t/s (**-16%**), for **zero** compute change — the kernel stays f16.

⇒ **The real prize needs no requantisation from us.** Experts are 92-97% of weights and
dominate prefill FLOPs, running f16 (25.51 TFLOPS) where int8 DPAS offers ~51 TOPS. An
int8 fused-MoE kernel could plausibly approach 2x prefill (~700 -> 1200+ t/s) and our
EXISTING u4 experts should pick it up automatically, since the u4->int8 machinery
demonstrably exists for FullyConnected. This is a plugin gap, not a model gap.

Also corrected: §11.1's claim that Qwen3.6 is intrinsically faster. Both configs are
byte-identical in architecture (`qwen3_5_moe`, 40 layers, hidden 2048, 256 experts,
top-8, 2 KV heads, head_dim 256). The +36% decode is ENTIRELY bit allocation: dense
path 1.78 GB (u4-heavy) vs 2.43 GB (all 8-bit) => bytes/token 2.32 vs 2.95 =>
ceilings 45.6 vs 35.9 t/s = **+27% predicted vs +36% measured** (rest within noise).
⇒ Ornith could recover most of it by requantising ITS dense path to int4
(`recompress_35b_text.py --mode int4`), keeping the fine-tune. The §11.2 quality
comparison is therefore CONFOUNDED — it varied checkpoint AND dense precision at once.

### 11.10 [R] Decode decay is 100% ACCEPTANCE COLLAPSE — the target is flat. §5.4 was half wrong.

§5.4 attributed long-generation decay to "growing-KV attention cost AND drafter
degradation". Measured properly on the 35B, varying only `--max-new` [M]:

| max_new | no draft | DFlash nat=8 | acceptance |
|---|---|---|---|
| 128 | 21.27 | **34.49** | 32.4% |
| 1,000 | 21.89 | **35.71** | 38.2% |
| 4,000 | 21.08 | **18.78** | **17.8%** |

**The target does NOT decay** — 21.27/21.89/21.08 is flat within noise. Growing-KV cost
is not a factor, and it never could have been: only 10 of 40 layers hold KV at ~15
KiB/token, so 4000 tokens is 60 MB against 2.95 GB read PER TOKEN (a 0.5% effect).
(An earlier partial run showed 21.35 -> 18.71 and I read it as real decay; it was
single-rep noise.)

**All decay is acceptance collapse.** A = 1+8*acc goes 4.06 -> 2.42 (−40%) while decode
goes 35.71 -> 18.78 (−47%); per-forward time grew only ~13%.

⇒ **PAST ~2-4k TOKENS THE DRAFT IS A NET LOSS**: 18.78 drafted vs 21.08 undrafted at
4000 tokens. DFlash is +62% on short replies and −11% on long ones. This is why the
15k-token aquarium generation limped at ~6 t/s — it ran a drafter that had stopped
paying for itself thousands of tokens earlier.

**Actionable: gate the draft on expected output length.** Short/interactive => draft on.
Long-form generation => draft OFF. Plain Qwen3.6 (§11.4) already runs draftless and so
is immune to this.

Likely cause of the collapse, consistent with the 9B logs showing
`draft=[279,279,279,279,279]` accepted=0: the drafter is **base-Qwen distilled, not
Ornith**, so the mismatch compounds as generated text drifts into the fine-tune's own
distribution. That makes an Ornith-distilled drafter more valuable than §5.2 implied --
not just +9 t/s at short length, but removing a cliff that currently makes speculation
harmful past a few thousand tokens.

### 11.11 [R] "Past 2-4k tokens the draft is a net loss" — RETRACTED. It was `ignore_eos=True` benchmark garbage.

§11.10 concluded acceptance collapses past ~2k generated tokens and recommended gating
the draft on output length. **Wrong.** Profiling the per-round `draft=`/`target=` token
streams (not guessing) shows what actually happens:

`bench_35b.py` sets `cfg.ignore_eos = True` to force exactly `--max-new` tokens. The
model FINISHES its answer around token ~1950 -- precisely where the "cliff" is -- then
keeps going, fabricating its own conversation:

```
target: 'Hello! 😊 How can I help you today?\nuser\n\nassistant\n<think>\nThe user has
         sent multiple empty messages. This is clearly a pattern of accidental inputs...'
draft:  'HelloHelloHelloHello can can can you'   truth=' Hello!'  accepted=1
draft:  ' \ufffd \ufffd can can can can can can'          truth=' 😊'     accepted=1
draft:  ' can can can today today today today'   truth=' How'    accepted=0
```

The drafter flails on synthetic multi-turn drift it was never going to predict —
including a PARTIAL EMOJI where the target emitted 😊, i.e. multi-byte token boundaries.
This is the same degenerate `[279,279,...]` pattern seen in the 9B logs.

⇒ **Decode does not degrade on real work.** Up to the natural end of the answer,
acceptance is ~36-42% and decode ~35 t/s. The "collapse" is an artifact of measuring
past the point the model would have stopped.
⇒ **Do NOT gate the draft on output length** (the §11.10 recommendation). There is
nothing to gate.
⇒ Any `ignore_eos=True` benchmark measures drafter performance on garbage once the
answer ends. For decode benchmarks, either keep generations shorter than the natural
answer or set `ignore_eos=False` and accept variable length.

STILL UNEXPLAINED: the aquarium runs used real EOS and genuinely produced 15k tokens at
~6 t/s. That is a legitimate long generation and is NOT explained by this artifact.
Hypotheses killed so far for that: growing-KV cost (target is flat at ~21 t/s at every
length, §11.10), total context length, and the drafter's 4096 sliding window (a
6198-token prompt gives 33-49% acceptance and 32.25 t/s, i.e. the BEST result -- so a
long prompt is fine and only long *generation* hurts).
