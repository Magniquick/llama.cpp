# ⏸ STOPPED 2026-08-03 — new Qwen generation announced

Halted deliberately: Qwen3.8-27B / Qwen3.8-Max were announced, so further work on
35B-A3B quantisation was poor timing. Nothing is running; both Kaggle builds are in ERROR.

**How far it got.** The last run cleared every gate we built:
```
pre-flight: HF_TOKEN present (len 37)          <- secret works (via Save Version only)
donor read, 85s                                 <- ofirzaf donor downloads and parses
[disk] 1082.9 GB free                           <- disk was never a constraint
542 quantised tensors to rebuild
pre-flight: donor coverage matches the validated plan   <- 542+191 confirmed on the REAL donor
TypeError: data type 'bfloat16' not understood
```

**The one remaining fix is written but never run.** `safe_open(..., framework="np")` cannot
materialise BF16 (numpy has no such dtype) and both source checkpoints are BF16. Switched to
`framework="pt"` with `.float().numpy()` in `kat_build_kernel.py`; the local proof that the
pt backend reads a real bf16 tensor was interrupted, so **that fix is UNVERIFIED**.

To resume: verify the bf16 read locally, regenerate `build-{ornith,kat}/`, push, then launch
with **Save Version in the editor** (a CLI `kernels push` produces a version that cannot read
secrets — `ConnectionError`, permanent for that run).

**Revoke the HF token** if not already done: fine-grained, write-scoped to the two target
repos, named `kaggle-opt1-build (revoke after)`.

**What transfers to any new model.** The pipeline is architecture-shaped, not model-shaped:
for any `qwen3_5_moe`-alike it needs only a config-identical single-file OV donor. The
allocation, the 8 fp transform rules, the donor/source validators and the numeric splice
tests all carry over. A *dense* 27B does not — measured arithmetic puts it at ~5-6 t/s here,
under the 20 t/s floor, because it reads ~15.6 GB/token against the MoE's ~2.4 GB.

---

# 35B coding-model quantisation — status (2026-08-03)

Goal: pick and build a quantisation for a 35B-A3B coding model on the Arc 140T iGPU.
Hard floor 20 t/s decode, quality maximised subject to it. Full evidence in
`KAT-QUANT-SWEEPS.md`; raw data in `kaggle-quant/sweeps/`; plots in `kaggle-quant/plots2/`.

## The chosen spec ("option 1")

| component | precision | why |
|---|---|---|
| dense (`in_proj_*`, `out_proj`, q/k/v/o, shared expert) | **i8 group-128 SYMMETRIC** | grouped 8-bit must be symmetric — see cliff below |
| routed experts | **u4 group-32 asymmetric** | finest the fused MoE kernel allows; zero-point mandatory |
| lm_head | **u8 per-channel asymmetric** | 0.39% vs 2.86% flips against u4_g32 |
| router, `in_proj_a`, `in_proj_b`, conv1d, norms | **f16/f32** | ~50 MB total, discrete-error risk |
| KV cache | **u8** | +18.8% at 16k, flat with context |

Measured whole-build: **5.34% top-1 flips (N=1536), ~21.3 t/s** no-draft, flat across
context. Current Ornith production (`armB-ABCD`) sits at ~9.64% flips / ~28 t/s.

## Findings that changed the design

1. **Experts dominate quality damage, not the dense path.** Experts alone at g64 cost
   5.53% flips (Ornith) / 5.86% (KAT) against 1.56% / 2.60% for the ENTIRE dense path at
   u8. **Replicated on both models.** The activation-error proxy said the opposite and was
   wrong; bytes were a bad proxy for damage.
2. **Three silent runtime cliffs**, all measured, none raising an error:
   - all-u4 MoE rule: `KeepMOE3GemmConstPrecision` matches only if *every* expert constant
     is u4. One u8 → weights decompress at compile time → **4.2×**, pure bandwidth.
   - grouped asymmetric 8-bit → falls off the fast FC kernel → **11×** (41.7 ms vs 3.6).
     Symmetric i8 grouped is fine (4.85 ms).
   - **"reports as compressed" ≠ "runs fast"** — `runtimePrecision == i8` was still
     reported in both broken cases. Timing is the only reliable check.
3. **OV folds a `+1` into the DECODER RMSNorm constants** (`input_layernorm`,
   `post_attention_layernorm` = `1 + w`), but NOT `linear_attn.norm`. Weight matrices are
   clean (cos 0.995 vs HF, zero offset).
4. **KAT vs Ornith is a tie.** Paired teacher-forced BPB over 100 MBPP+HumanEval items:
   KAT better on exactly 50/100, sign test p = 1.00. Quantisation costs <1% BPB on both.
   → no evidence for the model swap; the *quant* upgrade is the supported win.
5. **`DYNAMIC_QUANTIZATION_GROUP_SIZE` resolves to per-token on our IRs** because OV's
   GatedDeltaNet detector misses a decomposed export. Pinned to 128; costs ≤1%.
6. **Run-to-run variance is ~5%** — differences below that need repeats.

## Build tooling (validated, not yet run)

- `tools/kat_build.py --dry-run` — **PASSES**: donor→source mapping complete, every shape and
  source name probed against the real graph and checkpoint index.
- `kaggle-quant/kat_build_kernel.py` — all four branches implemented (dense, experts-down,
  experts gate/up, lm_head), fails loudly if any class is unrebuilt.
- Chain builder self-tested for all three topologies incl. 3-D stacked experts.
- Donor: `OpenVINO/Qwen3.6-35B-A3B-int4-ov` — config-identical to both models (0
  mismatches / 20 keys). Layout verified by measurement: dense cos 0.995;
  **`VariadicSplit.0` = gate** (cos +0.9952 vs −0.0099 for the wrong half).

## ⚠ STOP — the build is NOT ready. Two defects found 2026-08-03, both fatal.

The `--dry-run PASS` message is misleading: it only checks that *planned* tensors map,
never that the plan is *complete*. Do not launch the Kaggle build until both are fixed.

**1. The HF donor repo is the wrong artifact.** `kat_build_kernel.py` downloads
`OpenVINO/Qwen3.6-35B-A3B-int4-ov` and reads `openvino_model.xml`. That repo is a **VLM
export** — `openvino_language_model.*`, `openvino_text_embeddings_model.*`, plus a vision
tower — and contains **no `openvino_model.xml`**. The kernel would fail after downloading
~19 GB. The local `Qwen36-35B-int4-ov-cb` that the dry run validates against is a
*locally derived* CB merge (it even carries an `openvino_model.xml.bak`); it is not that
repo, and the 8 MB graph it depends on exists only on this box.

**2. Only quantised MatMul weights are transplanted — 523 MB stays Qwen3.6's.**
The rebuild loop only visits Constants of bitwidth 4/8 feeding a dequant chain, and the
`f16` plan class is `continue`d rather than replaced. Measured leftovers:

| tensor | count | size | consequence |
|---|---|---|---|
| `embed_tokens` (`self.weight`) | 1 | **508.6 MB** | Qwen3.6 input embeddings under source weights |
| `mlp.gate` (router) | 40 | 10.5 MB | Qwen3.6 routing into source experts |
| `in_proj_a` / `in_proj_b` | 60 | 2.0 MB | wrong GatedDeltaNet gating |
| `linear_attn` conv1d | 30 | 1.0 MB | wrong short-conv dynamics |
| RMSNorm (anonymous `Constant_NNNNN`) | ~110 | small | wrong per-layer scales |

`embed_tokens` is the killer, and it hides in plain sight: **two** constants have shape
`(248320, 2048)` — `self.model.lm_head.weight`, which the plan matches, and `self.weight`,
which it does not. With `tie_word_embeddings: False` they are independent, so the output
head is the source's while the input embeddings are the donor's. For KAT — which also
ships a *different tokenizer* — the token ids would index Qwen3.6 embedding rows. The
result would not be a quantisation artifact; it would be a different model, and the
terminal-bench score would have been read as quantisation damage.

### Progress 2026-08-03 (later): defect 1 solved, defect 2 all but the norms

**Donor: use `ofirzaf/Qwen3.6-35B-A3B-int4-ov`.** It is the only single-file OV export of
this model on the Hub (`openvino_model.xml` + `.bin`), and its `openvino_model.bin` is
19,155,614,274 bytes — byte-identical in size to the local CB donor, so almost certainly
the same quantisation. Two things fall out:
- it is downloadable on Kaggle, which the VLM repo never was;
- it **names** the embedding tensor `self.model.model.embed_tokens.weight`, where the
  local CB donor leaves it anonymous as `self.weight`.

Its naming omits the `language_model.` segment, but the plan regexes match on substrings,
so the counts come out identical (dense 250, f16 100, experts_gate_up 80, experts 40,
lm_head 1). `plan_for()` now handles both, matching embed by name and falling back to
shape `(248320, *)` for the anonymous case.

Plan extended and re-validated against the local donor — **542 mapped, 0 unmapped,
0 missing**: `embed` 1, `dense` 250, `f16` 140, `conv1d` 30, `experts` 40,
`experts_gate_up` 80, `lm_head` 1. The `f16` class now *rebuilds from source* instead of
being skipped (it had silently kept the donor's router and GDN gates), and picked up
`shared_expert_gate`; `conv1d` is matched via its `aten::_convolution` consumer since it
has no weight name.

**Norms: solved, 81/81.** They are anonymous, so they resolve through their *consumer's*
name (`layers.N.input_layernorm/…`, `…post_attention_layernorm/…`, `language_model.norm/`).
The `1 + w` fold was verified rather than assumed, by range-fetching
`Qwen/Qwen3.6-35B-A3B` and comparing to the donor's actual values:

| tensor | HF `w` | donor const | |
|---|---|---|---|
| `input_layernorm` | +0.0311 | +1.0311 | `1+w` |
| `post_attention_layernorm` | −0.1049 | +0.8951 | `1+w` |
| **final `norm`** | +1.6279 | +2.6279 | `1+w` |

That **extends the previously documented finding**, which covered only the two decoder
norms — the final norm folds too. `linear_attn.norm` does not (checked separately, stored
raw), so it must not get the +1.

**All 191 fp constants now resolved**, via consumer name, with a **per-class** transform.
Every rule was established by comparing the donor's actual values against range-fetched
`Qwen/Qwen3.6-35B-A3B` tensors — the transforms are *not* uniform, and assuming they were
would have silently corrupted three of the eight classes:

| transform | tensors | count |
|---|---|---|
| `1 + w` | input_layernorm, post_attention_layernorm, final norm, **q_norm, k_norm** | 101 |
| raw `w` | linear_attn.norm, **dt_bias** | 60 |
| **`exp(w)`** | **A_log** — donor stores `exp(A_log)`; the graph negates downstream, which is why the consumer is `aten::neg`. Writing raw `w` would have looked entirely plausible. | 30 |

Counts reconcile: 40+40+1+10+10 = 101, 30+30 = 60, 30. `rotary_emb` is the one genuinely
generic constant (derived from `rope_theta`/`partial_rotary_factor`, already proven
config-identical), so inheriting it is correct rather than merely tolerable.

Dry run is now **PASS** on 542 quantised + 191 fp, 0 unmapped, 0 missing — and this time
the gate looks at everything ≥8 elements of any dtype.

**Three times the gate passed by only looking where it already knew to look** — bitwidth
4/8 only, then fp ≥2048 only, then the right scope. "0 unmapped" is a statement about the
scan's scope, not about the model; each widening found real bugs.

### Kernel ported; donor still unresolved

`kat_build_kernel.py` now carries the full plan. Its `FP_RULES` table is **byte-identical**
to the validator's (checked by AST diff), `resolve_fp` passes 8/8 on real consumer names,
`apply_fold` reproduces the measured donor values for all three transforms, and rule
ordering is verified so `linear_attn.norm` is not shadowed by the `linear_attn/` rules.
The `f16`/`conv1d`/`embed` branches and a second pass over the 191 fp constants are
implemented.

**Donor resolved: `ofirzaf/Qwen3.6-35B-A3B-int4-ov` is a validated drop-in.**

| | local CB | ofirzaf |
|---|---|---|
| quantised | 542 | **542** |
| fp | 191 | **191** |
| unmapped / unresolved | 0 / 0 | **0 / 0** |

One real bug came out of the comparison: ofirzaf names the final norm `.model.norm/` where
the local donor has `.language_model.norm/`. The rule now accepts both, verified not to
catch `linear_attn.norm` or `q_norm`.

*Correction to an earlier entry here, which claimed ofirzaf was missing `embed_tokens` and
one `gdn_norm`.* That was a bug in my XML validator, not in the donor: I keyed layers by
`id`, which is unique only **within** a graph body. Both donors carry 31 bodies — the main
graph plus 30 GDN recurrence loops — so ids collided and entries silently overwrote each
other. `self.model.model.embed_tokens.weight [248320, 2048] i8` is present. Walking each
`<layers>`/`<edges>` pair separately gives the table above.

The nested bodies are structurally identical between the two donors and contain only
scalar control constants (`i32 []`, `boolean[1]`), no weights — so OV's `get_ops()` not
descending into subgraphs costs nothing here.

A **pre-flight coverage check** runs immediately after planning and *before any shard is
fetched*, raising unless the donor yields exactly 542 + 191. The old completeness check ran
after the rebuild loop — hours of downloading later — and only knew about the classes it
already had.

### An unquantised replacement must swap the chain TAIL, not the leaf

The first version of the `f16`/`conv1d` branch (my own, caught on review) did:

```python
cst = op.constant(w.astype(np.float16), ov.Type.f16)
new_out = op.convert(cst, node.get_output_element_type(0))   # f16 -> u4  ← destroys it
```

The donor stores these as a full dequant chain —
`Constant(u4) → Convert → Subtract(zp) → Multiply(scale) → Reshape → Convert(f32) → MatMul`
— so casting float weights to 4-bit integers *and then* applying the donor's zero-point and
scale on top would have silently destroyed exactly the small model-specific tensors the
transplant exists to fix.

`dequant_tail()` now walks the chain and replaces its **output**. It matches on **element
count, not rank**, because the grouped form is rank 3/4, the Reshape restores the logical
rank, and conv1d's tail is `(8192,1,1,4)` while the checkpoint stores it 3-D — counting
elements handles all three without special cases, and a genuine layout disagreement returns
`None` and raises instead of corrupting. Verified against the real graph, 0 failures:

**The same bug was in the QUANTISED path too**, found by auditing the rest of the loop
rather than by a test. `build_chain()` returns a *complete* chain
(`Const → Convert → [Subtract] → Multiply → Reshape`) already in logical shape, yet the
loop spliced it into the **leaf's** consumer — pushing dequantised values back through the
donor's own `Convert/Subtract(zp)/Multiply(scale)`. All 542 quantised tensors. Now spliced
at the tail as well.

`dequant_tail()` **prefers the last f16 node**, not the literal end of the chain. The donor
ends `… → Multiply(f16) → Reshape(f16) → Convert(f32) → MatMul`; splicing at the f32
Convert would force `build_chain` to emit f32, which mismatches the MatMul's other operand
(caught by a synthetic-graph test) *and* changes the dequant pattern the MoE matcher keys
on — a missed `KeepMOE3GemmConstPrecision` being the silent 4.2× cliff. Stopping at the
last f16 node leaves the donor's own Convert intact.

Verified on a synthetic graph with the donor's exact chain: splice changes the output and
reproduces the source weights to **0.73% relative error** — exactly i8 g128 symmetric
quantisation error, i.e. the graph now computes the source weights rather than the donor's.
All **542 tails resolve, dtype f16, 0 missing**:

| class | count | class | count |
|---|---|---|---|
| `dense` | 250 | `f16` (router/gates) | 140 |
| `experts_gate_up` | 80 | `experts` | 40 |
| `conv1d` | 30 | `embed` / `lm_head` | 1 / 1 |

### Numeric verification of the splice (all three geometries)

Count-based checks stayed green through both leaf-vs-tail bugs, because they measure
coverage, not whether the spliced graph computes the right thing. So each geometry is now
checked by building a synthetic graph with the donor's exact chain, splicing through the
**real kernel helpers**, and comparing against an exact matmul on the source weights:

| geometry | scheme | rel. error | expected |
|---|---|---|---|
| 2-D dense | i8 g128 symmetric | **0.0073** | matches the sweeps' dense error |
| 3-D stacked experts | u4 g32 asymmetric | **0.0547** | matches the ~5–8% expert error |
| 2-D per-channel (embed/lm_head) | u8 per-channel | **0.0067** | matches lm_head 0.78% |

Each also asserts the output *changed* versus the donor — otherwise a no-op splice would
pass silently. The per-channel case additionally asserts the chain stays **rank 2**: a
rank-3 result is treated by MatMul as *batched* rather than erroring, which is the historic
4526% bug.

The remaining two branches are now covered as well, so **all five paths are verified**:

| path | check | result |
|---|---|---|
| `f16` / `conv1d` | plain constant spliced at the tail | **0.00032** — pure f16 rounding, ~20× tighter than the quantised paths, which is itself the evidence that no quantisation is applied |
| fp constants | `1+w` fold on an RMSNorm-shaped constant | **exact** (0.00e+00); `1+w` / `w` / `exp` all correct |

The f16 branch's error being an order of magnitude *lower* than the quantised paths is the
useful signal — if it had accidentally gone through a quantiser it would read ~0.007, not
~0.0003.

## Blocked / next

1. **BLOCKER: add `HF_TOKEN` (write scope) as a Kaggle secret** (Add-ons → Secrets).
   The build output is 21.20 GB, over Kaggle's ~20 GB cap, so it must push straight to HF.
   Dropping experts to g64 would fit (19.94 GB) but costs quality (5.53% vs 4.36%).
   Push path is now implemented (`publish()`): token via `UserSecretsClient`, never in
   source; copies the donor's tokenizer/config alongside the IR and writes a model card.
2. Build Ornith@opt1 and KAT@opt1 on Kaggle → push both to HF (`Magniquick/*-opt1-ov`,
   private repo already created) → pull ~42 GB (~3.4 h).
3. **TerminalBench locally**, not on Kaggle: OV has no Gaudi/NVIDIA plugin, so Kaggle
   would run CPU-only at single-digit t/s. The Arc is our fastest device for these IRs.
   Harness is ready and fixed — run with `run/tb-eval.sh <model-id> [n-tasks] [port]`.

## Harness fixes (2026-08-03)

- **Rootless podman had NO `unqualified-search-registries`** — every bare Docker Hub image
  name failed to resolve, taking the whole task down with `docker compose up` exit 125
  and a useless `unknown_agent_error`. This was not minio-specific; it would have hit a
  large fraction of tasks. Fixed in `~/.config/containers/registries.conf`
  (`docker.io` + `short-name-mode = "permissive"`, since the harness has no tty).
  Verified: the task that failed now brings up all 4 containers.
- **`podman-compose down` does not force-stop**, so killed or timed-out tasks leak their
  containers — two were still up from the aborted trial. On a 30.84 GB shared-memory box
  with a ~21 GB model that risks an OOM mid-run. `tb-eval.sh` reaps on entry and on trap.
- **Build kernel wrote to `/kaggle/working`** (~20 GB cap) for a 21.2 GB IR — it would
  have died at ~95% of the save, hours in. Moved to `/kaggle/temp`, plus a pre-flight
  free-space check that fails immediately rather than at the end.
- **terminus needs enforced structured output, and we could not do it.** `-a terminus`
  resolves to terminus_1, which sends `response_format=CommandBatchResponse` and then
  hard-parses with `model_validate_json`; `ov_serve.py` silently ignored the field.
  Measured against the incumbent: the model answers with **prose reasoning before the
  JSON**, so all three of terminus's retries fail and the task scores as an agent error.
  This would have produced a near-zero score for both models that reflects output
  formatting, not reasoning. Two changes:
  - `ov_serve.py` now implements `response_format` (`json_schema` / `json_object`) via
    `GenerationConfig.structured_output_config`, and degrades instead of 400-ing when
    the backend is missing (`unfence()` strips a ```json wrapper; unit-tested).
    Fence-stripping alone does **not** fix the prose preamble — enforcement is required.
  - The patched GenAI was built with `ENABLE_XGRAMMAR=OFF`, so the backend is
    unregistered. **Rebuilt into `genai-src/build-xg` — done, 394/394, rc=0** (separate
    dir: the working build is ABI-bound to `libopenvino.so.2640` and every runner
    depends on it). Opt in per-process with `OV_GENAI_BUILD=genai-src/build-xg`
    (new env override in `ov_boot.py`; the ABI pre-flight still applies). The default
    is unchanged, so nothing else can regress.
    Verified end-to-end on the GPU: backend registered, terminus's exact schema
    accepted, reply parses with `CommandBatchResponse.model_validate_json`, and it is
    **faster** — 13.7s vs 30.4s, because constraining removes the prose preamble.
    Flags recovered from the original `CMakeCache.txt`, not guessed: `BUILD_TOKENIZERS=OFF`
    (the `thirdparty/openvino_tokenizers` submodule is empty) and
    `Python3_EXECUTABLE=<venv>/bin/python3.12` (otherwise the extension builds
    cpython-311 and `ov_boot.py` refuses to load it).
  - Fallback if the rebuild fails: `-a terminus-2`, whose plain JSON/XML parsers extract
    from free text and tolerate a preamble. Not needed — the rebuild worked.
  - **Validated in a real terminus run** (`cartpole-rl-training`, incumbent, harness
    smoke test only — not a score): 10/10 episodes returned valid JSON, zero ParseErrors.

### Sizing the real run (settled)

The smoke test finished `agent_timeout`, 0.00% — **not** a parse failure. Per-task limits
live in `task.yaml` (`max_agent_timeout_sec`, 360 s for 56 of 80 tasks) and are tuned for
frontier APIs running 10–30× our decode rate. Unscaled, everything returns
`agent_timeout` and the scores carry no signal.

`tb-eval.sh` is now fixed to **5 easy tasks at `--global-timeout-multiplier 2`**:
- `--n-tasks N` is never used — N takes whatever sorts first, which is how a one-task
  smoke test landed on `cartpole-rl-training` (difficulty: **hard**). Task ids are explicit
  and identical across models.
- Only 12 of 80 tasks are `easy`; at ~69 s mean per episode that is the only tier
  finishable in budget.
- Worst case 1740 s × 2 = 58 min, inside the 1 h/model cap. Sized deliberately: if the
  3600 s cap truncated one model at 5 tasks and the other at 4, the comparison would be
  invalid.
- The multiplier applies to both models equally, and both are 35B-A3B at the identical
  opt-1 spec so their t/s is essentially the same — it widens dynamic range rather than
  favouring either.

**Statistical power is very low and the result should be read as a sanity check, not a
verdict.** Five binary outcomes cannot separate two models that already tied at 50/100
(p = 1.00) on paired BPB. What it *can* catch is a degenerate build — a model that loops,
emits broken commands, or fails to use the terminal at all.

One measurement gotcha: `/v1/models` can take >3 s to answer while the GPU is mid-
generation (single-flight), so a short-timeout health probe reports a live server as
down. Do not infer a crash from it — check the process.

### Port 80 is DPI-blocked here, and it silently scored a correct answer as a failure

Calibrating on `hello-world` exposed the last trap. The model **solved it in 14.8 s**
(`echo "Hello, world!" > hello.txt`, verified with `cat`) — then the run was scored
`test_timeout`, 0.00%.

Cause: every task's `tests/setup-uv-pytest.sh` opens with `apt-get update` over
`http://`. On this network port 80 is DPI-blocked — the TCP connection opens, the GET
goes out, and nothing ever comes back — while HTTPS to the same host answers in 0.2 s
(the zapret setup here covers 443 only). So the test phase hung for its full 120 s.
`curl` and `uv` are both absent from the base image, so apt genuinely is required.

Fix: `tb-eval.sh:patch_dataset()` prepends an https rewrite of the apt sources to all 80
`setup-uv-pytest.sh` files. Same `apt-get update` then takes **3.4 s**. Only dependency
installation is touched — the tests themselves are unchanged and the patch applies to
every model equally. Idempotent, and re-run on every invocation because re-downloading
the dataset would silently revert it.

**Verified: `hello-world` now resolves, 100.00%, 224 s end-to-end** (including ~50 s of
model load). The whole chain — podman → terminus → xgrammar-constrained JSON → agent →
tests — is working.

Measured headroom: ~170 s per easy task once loaded, so the 5-task set should run in
~15 min against a 58 min worst case. There is room to widen the subset for more
statistical power if the 1 h/model budget is allowed to grow — that is a judgement call
about power vs. budget, so it is left as the user's.
4. Free wins already applied to the runners: u8 KV, `nat=6` (+9% over the probe default),
   DQ pinned to 128.

## The donor supplies the GRAPH, not the TOKENIZER

`publish()` originally copied everything non-IR from the donor snapshot, tokenizer
included. Measured file sizes say that is wrong for KAT:

| repo | `tokenizer.json` | `tokenizer_config.json` |
|---|---|---|
| donor (Qwen3.6) | 19,989,343 | 1,139 |
| Ornith | 19,989,325 | 1,165 |
| **KAT** | **12,807,982** | **16,718** |

Ornith's tokenizer matches the donor to within 18 bytes and would have survived on luck.
KAT's is a different tokenizer entirely, and its `tokenizer_config.json` is 15× larger —
that is KAT's agentic chat template. Publishing the donor's would mis-tokenise every
prompt *and* format it with Qwen's template, which is precisely what terminal-bench
exercises.

Fixed: tokenizer, chat template and generation config now come from the **source** repo,
and `openvino_tokenizer/detokenizer` are **regenerated** from it — copying the donor's
compiled tokenizer IR would reintroduce the same bug in a form much harder to spot.
`config.json` still tracks the donor, since that is what describes the graph actually
built. Verified offline with a stubbed HF API (provenance + no-clobber assertions).

## Runbook, once the two IRs are on disk

Both models are already wired into the runners (`ov_serve.py` MODELS + `ornith-run.sh`
aliases, `MEM=29G` — 21.2 GB of weights vs the incumbent's ~19.5 makes 28G the floor,
not a margin). Expected local dirs: `ornith-35b-opt1-ov`, `kat-35b-opt1-ov`.
Verified by dispatch: both resolve to the right target and fail only on the absent files.

```sh
# per model -- the xgrammar build is REQUIRED (see above)
OV_GENAI_BUILD=genai-src/build-xg run/ornith-run.sh ornith-opt1 server --port 8090
run/tb-eval.sh ornith-opt1            # 5 easy tasks, timeout x2, ~15 min
# then stop the scope, repeat with kat-opt1
```

## Corrections made during the work (kept for calibration)

N=159 calibration set → 1536/4096; reference-cancellation bug twice (lm_head, then
tracks); summed component KLDs replaced by measured combined (sub-additive, ~0.88×);
bandwidth model mis-calibrated by pairing arm B's bytes with ABCD's throughput; the
"dyn-quant is skipped at batch ≤64" claim is FALSE (it fires at batch 1); a silently
failed `cp` that plotted stale data. Roughly half the "this changes the answer" moments
were my own bugs — the underlying decision converged early and stayed put.
