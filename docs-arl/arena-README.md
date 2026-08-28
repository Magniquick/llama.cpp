# llms/ arena — one strict ranking

Goal: decide which model in `~/Documents/llms/` is actually the best one to use, from tasks
derived from `~/Projects` and the GitHub repos, graded objectively rather than by vibes.

## Contestants

| key | model | notes |
|---|---|---|
| `ornith35` | Ornith-1.0-35B-A3B, unsloth UD-IQ3_XXS + MTP graft | **experts forced to CPU** — see below |
| `ornith9` | Ornith-1.0-9B, unsloth UD-Q6_K_XL, MTP fused | |
| `gemma12` | Gemma-4-12B-it QAT Q4_0 + q8 MTP draft | |

Sampling follows each model's published recommendation, not a house style:
Ornith `temp 0.6 / top_p 0.95 / top_k 20` (deepreinforce model card), Gemma
`temp 1.0 / top_p 0.95 / top_k 64` (Google's `generation_config.json`, also in the GGUF
metadata as `general.sampling.*`). Ornith runs through the froggeric fixed Qwen template.
Reasoning traces are separated into `reasoning_content` (`--reasoning-format deepseek`)
so that only the model's real answer is graded.

### The 35B is crippled by a driver bug, not by the benchmark

Ornith-35B UD-IQ3_XXS throws `vk::Device::waitForFences: ErrorDeviceLost` on the first
decode — on the packaged `llama.cpp b10068` *and* on the pinned source build, on the
pristine unsloth file as well as the MTP-grafted one, at 2k context as well as 32k.
Everything else was ruled out (`logs/rescue35.txt`, `logs/rescue35b.txt`):

| config | expert tensors on Vulkan | result |
|---|---|---|
| `-ngl 99` | all 41 layers | ErrorDeviceLost |
| `--n-cpu-moe 1` / `20` / `36` | 40 / 21 / 5 layers | ErrorDeviceLost |
| `--cpu-moe` | none | **works**, 6.7 t/s, 78% MTP acceptance |
| `GGML_VK_DISABLE_{MMVQ,INTEGER_DOT_PRODUCT,DOT2,COOPMAT,COOPMAT2,F16,FUSION}` | all | ErrorDeviceLost |

`--n-cpu-moe 1` leaves essentially the same byte count in the Vulkan heap as the crashing
config, so this is not allocation pressure — it is the IQ3_XXS expert matmul shader. Only
taking *every* expert tensor off the Vulkan backend survives. So the 35B runs at ~6.7 t/s
instead of ~20. **This affects its speed column only.** Where a kernel executes does not
change the sampled distribution, so its quality scores stay comparable.

## Method

Two harnesses, both objective-first.

**Single-shot code tasks** (`tasks/`, 8 tasks). Each is a precise spec drawn from real code
in `~/Projects`; the model emits one file; hidden tests it never sees are then run against
it in a sealed sandbox. Score = fraction of hidden assertions passed, so a near-miss scores
above a non-starter and a plausible-looking non-compiling answer scores zero.

| task | source | graded by |
|---|---|---|
| `r1_dpi_newtype` | `mouse-dpid` DPI validation | rustc + 14 tests |
| `r2_hid_frame` | `mouse-dpid` BLE vendor HID framing, 0x55 checksum | rustc + 9 tests |
| `r3_noise_floor` | `ultralurk` detector, incl. the real floor-collapse latch bug | rustc + 15 tests |
| `c1_msr_bits` | `DisablePROCHOT` + `yoga-powerd` RAPL MSR encoding | gcc + ~50 assertions |
| `p1_rbm_free_energy` | `nitc` RBM-on-Ising, log-sum-exp stability | numpy + 29 checks |
| `d1_date_bisect_debug` | `sha1-hulud-dumper` 1000-result API cap | 24 checks incl. a perf bound |
| `s1_hardlink_dedupe` | phone-backup dedupe under `~/backup` | 32 checks on a real fixture tree |
| `w1_html_tool` | single-file fan-curve editor | jsdom, driven headlessly |

**Multi-turn agentic** (`tasks_agentic/`, 2 tasks). The model gets real tools
(`list_files`, `read_file`, `write_file`, `run_tests`, `finish`) over the OpenAI `tools`
field and is driven in a loop against a broken project with **two bugs in two different
files**. Grading is on the final state of the work directory against held-back tests, plus
process metrics only a multi-turn run reveals: turns used, malformed tool-call JSON, bad
paths, redundant re-reads, whether it ever ran the tests, whether it declared victory
falsely. The visible test file is restored from a pristine copy before grading, so editing
the tests cannot buy a score (and doing so is recorded as `edited_tests`).

| agentic task | shape | broken baseline → correct |
|---|---|---|
| `a1_fix_fan_curve` | two independent bugs in two files, found from test output alone | 0.519 → 1.000 |
| `a2_trace_config_bug` | four bugs in one merge layer, only diagnosable by reading a `SPEC.md` that contradicts the code | 0.676 → 1.000 |

## Ranking policy: lexicographic, agentic breaks ties

Several code tasks saturate at 1.00 for more than one model, so a mean alone does not give a
strict order. The sort is therefore lexicographic, not blended:

1. code correctness (mean over the 8 hidden-test tasks)
2. **agentic score** — the tie-break
3. false-victory penalty: called `finish` while the held-back tests still failed
4. wasted tool actions: malformed JSON + bad paths + bad args + unknown tools + redundant
   re-reads + turns that produced no tool call
5. turns taken to get there
6. speed, last

A blended average was rejected because it lets a 0.02 code difference outvote a total
agentic failure, which is the opposite of what a tie-break is for.

## Token budget

Code tasks get 9000 tokens. A reasoning model can spend that entire budget thinking and
emit **no answer at all**, and scoring that 0 would measure the budget rather than the
model — on an 8-task suite one such zero moves the headline by 12.5% and can flip the
order. So a response that is truncated *with an empty answer* gets exactly one retry at 2.5x
budget, flagged `retried_after_truncation`; verbosity still shows up as cost in the
tok/task and s/task columns. This was not hypothetical: Gemma hit it on
`s1_hardlink_dedupe`, needed 11259 tokens, and scored 0.594 instead of 0.000.

## Measured throughput (MTP on)

| model | decode | prefill | MTP acceptance |
|---|---|---|---|
| gemma12 | 15.4 t/s (13.3–16.5) | 111–136 t/s | 77% |
| ornith9 | 14.8 t/s (13.6–16.0) | 115–127 t/s | 68–79% |
| ornith35 | 6.7 t/s | — | 78% |

Short generations flatter the models: Gemma probes at 25 t/s over 400 tokens but settles to
14.4 t/s over 9000, because decode slows as the KV cache fills. Decode is bandwidth-bound —
Ornith-9B reads 8.8 GB/token for 14.8 t/s, implying ~70 GB/s effective, which matches the
STREAM measurement. Hitting ~50 t/s on this box needs ≤~1.2 GB touched per token, which
only a low-active MoE can do: the 35B is an A3B and *should* be the fastest model here, and
is only the slowest because the expert-shader fault forces its experts onto the CPU.

### Websites are judged on behaviour, never on looks

`w1_html_tool` is loaded in jsdom and *operated*: inputs get real `input` events, buttons
get real clicks, and the grader asserts on interpolation arithmetic, live recomputation,
validation edge cases (empty string, `abc`, out-of-range, non-monotonic), add/remove
bounds, and sort ordering. Any `console.error` or uncaught exception fails the run. No
judgement is made about styling, which nobody in this pipeline can see.

### Guardrails

* Every hidden test suite is validated against a reference solution I wrote first
  (`ref/`, `bin/validate.sh`) — all 8 score exactly 1.0, so no model is graded against an
  unsatisfiable bar. This caught two bugs in my own specs and one in a reference.
* Model-written code executes under `bwrap` with the filesystem read-only apart from the
  work directory, and no network.
* `llama-server` runs inside a `systemd-run` scope with `MemoryMax=22G`,
  `MemorySwapMax=0`, plus `--no-mmap`. On an iGPU the Vulkan heap is host RAM, so an
  mmap'd model gets charged twice; a 16.9 GB model reached 33 GB total-vm and triggered a
  global OOM that killed the desktop session. The scope confines a repeat to the server.
* Three samples per (model, task), aggregated as the mean. Tight specs make models
  regurgitate near-identical memorised answers; the reported `spread` column shows how much
  each model swings between samples on the same task.

## Layout

```
models.json            contestants + sampling + working flags
tasks/<id>/            PROMPT.md, meta.json, grade.sh, assets/hidden_tests.*
tasks_agentic/<id>/    PROMPT.md, fixture/, hidden/, verify.sh
ref/<id>/              my reference solution (the bar every hidden suite is validated against)
runs/<model>/<task>/s<N>/          prompt, content, reasoning, response.json, files/, grade.json
runs_agentic/<model>/<task>/s<N>/  transcript.json (every tool call), work/ (final tree), grade.json
bin/                   serve, run_arena, agentic, score, validate, fitcheck, rescue35*
logs/                  server logs, rescue matrices, overnight.log
ranking.json           machine-readable final ranking
```

Every generation is preserved verbatim, including the HTML artifacts
(`runs/<model>/w1_html_tool/s<N>/files/index.html`) and the agentic work trees.

## Re-running

```bash
bash bin/validate.sh                     # prove the graders still pass on the references
bash bin/overnight.sh                    # full sweep, resumable, pass-ordered
python3 bin/score.py                     # ranking from whatever has finished
python3 bin/fitcheck.py --ctx 102400     # does a model fit a given context, measured
```
