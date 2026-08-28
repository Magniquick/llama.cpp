# Speculative decoding corrupts hybrid (GatedDeltaNet / linear-attention) models in OpenVINO GenAI

Source read: `openvino.genai` @ `fbbeb800ff5e6c6029b3747f87c1231a8b424d5e` (master, version
`2026.4.0.0` in `CMakeLists.txt`; **no `2026.4.*` tag exists yet** — latest tag is `2026.2.1.0`).
Clone used for this analysis: `/home/magni/Documents/llms/genai-read`
(identical commit to the build tree `genai-src`).

All line numbers below refer to that commit, paths relative to `src/cpp/src/`.

---

## 1. Root cause

For a hybrid model (`qwen3_5`: 24 GatedDeltaNet `linear_attention` layers + 8 `full_attention`
layers) on a non-NPU device, `LLMPipeline(..., draft_model(...))` does **not** use the stateful
speculative pipeline — `llm/pipeline.cpp:40-56` (`should_use_stateful_pipeline`) returns `false`
unless the device is NPU or the pair is Gemma4-MTP, so execution lands in the **Continuous
Batching / PagedAttention speculative path**
(`ContinuousBatchingPipeline::SpeculativeDecodingImpl`,
`speculative_decoding/continuous_batching/fast_draft_strategy.cpp`). That path treats *all* model
state as a token-addressable paged KV cache. It rewinds a request purely by decrementing the
logical token counter (`update_processed_tokens_num`) and letting the next pass recompute the KV
entries for those positions. That is valid for KV blocks (each token owns its own slot) and
**invalid for GatedDeltaNet conv/recurrent state**, which is a single fixed-size block mutated
multiplicatively in place: `scheduler.hpp:829-835` emits
`block_indices = [block, block]`, `cache_interval = 0` for every linear-attention sequence, i.e.
*read block == write block*. The recurrent state after the verify pass therefore reflects
`seed + all K drafted tokens`, and there is no copy of the state at the accepted prefix. The
moment one drafted token is rejected, the committed state is permanently ahead of the committed
token sequence, and — because the recurrence is multiplicative — the error compounds on every
subsequent step. That is exactly the observed behaviour: coherent until the first rejection
(~char 62), then unrecoverable degeneration into `2. 2. 1. 1.`.

The hypothesis in the task brief is **confirmed**, with one correction: it is *not* true that
corruption happens at 100 % acceptance. Nothing in the code over-advances the state when every
candidate is accepted (`past_length = P`, `K+1` scheduled tokens, `num_processed_tokens` ends at
`P+K+1` — consistent). The self-draft experiment does not actually run at 100 % acceptance: the
draft samples tokens one at a time through the *decode* GDN path while the target verifies
`K+1` tokens in one *chunked* GDN pass, and those two paths differ by exactly the float noise
already measured on this machine (max logit delta 0.015). Any near-tie argmax flips, one
candidate is rejected, one rollback happens — and a single rollback is fatal and permanent. So
self-draft corrupting is consistent with (indeed *predicted by*) rejection-triggered rollback; it
is not evidence of a second bug. There is also no `IVariableState` / `query_state()` handling
anywhere in the CB path at all: under PagedAttention the GDN state lives in externally-managed
cache tensors (`CacheType::LINEAR_ATTENTION_CACHE`), not in `ov::IVariableState`, so
`query_state()` is irrelevant here (it *is* the mechanism in the stateful path — see §3).

The infrastructure to do this correctly **already exists** and is used by exactly one strategy
(DFlash). Generic FastDraft and CB-Eagle3 simply never call it.

## 2. Exact code locations

### The bug

| What | Where |
|---|---|
| In-place LA state: read block == write block, `cache_interval = 0` | `continuous_batching/scheduler.hpp:829-835` |
| Main-model rollback after rejection (LA state left over-advanced) | `sampling/sampler.cpp:1774-1780` — `sequence_group->update_processed_tokens_num(min_processed_tokens)` |
| Draft-model rollback after rejection (same defect, draft side) | `speculative_decoding/continuous_batching/pipeline_impl.cpp:443-451`, esp. `:449` `request->update_processed_tokens_num(num_processed_tokens - result.removed_tokens_cnt + 1)` |
| Generic CB speculative round — **no** LA reserve/promote anywhere | `speculative_decoding/continuous_batching/fast_draft_strategy.cpp:181-271` (`SpeculativeDecodingImpl::step()`); candidates pushed at `:204-207`, target verified at `:216`, results read at `:222-226` |
| No guard either: `fast_draft_strategy.cpp` / CB `eagle3_strategy.cpp` never mention `linear_attention` | (grep: only `dflash_strategy*` and `pipeline_impl*` do) |

### The existing, correct mechanism (DFlash only)

| What | Where |
|---|---|
| Reserve N+1 per-token LA checkpoint blocks before the verify step | `dflash_strategy.cpp:501-503` → `pipeline_impl.cpp:547-565` → `scheduler.hpp:303-309` |
| Scheduler emits `[committed, tmp1..tmpN]`, `cache_interval = 1` (state snapshot after **every** scheduled token) | `scheduler.hpp:802-826` |
| Promote the snapshot matching the accepted prefix | `dflash_strategy.cpp:551-557` → `scheduler.hpp:311-317` → `block_manager.hpp:988-1020` (**pointer swap, zero copy**) |
| Slot arithmetic: `slot = accepted + 1` (verify input includes the seed token) | `dflash_strategy_utils.hpp:209-226` |
| LA pool sizing: `num_assistant_tokens + 2` blocks | `dflash_strategy_utils.hpp:139-152`, applied `dflash_strategy.cpp:300-304` |
| Release on early-exit / exception | `dflash_strategy.cpp:509-517, 527, 540-548, 607-618` |
| Model-runner plumbing is already generic over `block_indices.size()` | `continuous_batching/model_runner.hpp:1753-1833` |

## 3. Is `gemma4_mtp` a separate pipeline with correct hybrid handling? No.

`StatefulGemma4MTPLLMPipeline` (`speculative_decoding/stateful/gemma4_mtp_strategy.cpp`) *is* a
fully bespoke pipeline (own target/assistant wrappers, shared-KV outputs `mtp_full_attention_key`
etc., own `crop_state_to_length`), and it is selected before every other speculative path
(`llm/pipeline.cpp:196-198`). But it is **not** a template for hybrid state:

- `gemma4_mtp_strategy.cpp:80-82` — `OPENVINO_ASSERT(!m_cache_types.has_linear(), "Gemma4 MTP stateful speculative decoding does not support linear attention states.")`
- `gemma4_mtp_strategy.cpp:183-193` — rollback is `utils::trim_kv_cache`, i.e. slice the seq-len axis of every `query_state()` tensor; only meaningful for rank-4 KV.

The same guard exists in the other two stateful strategies:
`stateful/fast_draft_strategy.cpp:55-59` (*"KV cache rollback would reset the entire state instead
of trimming"*) and `stateful/eagle3_strategy.cpp:401-403`. And `utils.cpp:551-600` /
`utils.hpp:222-228` encode the global rule: linear state + any trim ⇒ `needs_reset()` ⇒ **reset
the whole state** (`utils.cpp:571` asserts a linear model never reaches the partial-trim path).

So the *only* place in the codebase that rolls back recurrent state correctly is CB DFlash. If you
want a `qwen35_mtp` strategy, model it on `speculative_decoding/continuous_batching/dflash_strategy.cpp`,
**not** on `gemma4_mtp_strategy.cpp`. DFlash also sidesteps the draft-side problem entirely: its
draft is a single grafted MTP head driven by target hidden states (`DFlashCBDraftRunner`), with no
CB pipeline and no LA cache of its own.

Detection of hybrid-ness works fine on this IR: `utils.cpp:478-512` classifies rank-3 /
one-dynamic-axis ReadValue inits as conv state and rank-4 / one-dynamic / zero-zero-axis as SSM
state — which is why the stateful paths *would* have thrown, and why the CB path (no such check)
is the one that silently corrupts.

## 4. Proposed patch

Three parts. Part A is the real fix for the target model and is a near-mechanical port of DFlash.
Part B closes the draft-side hole. Part C is the one-line safety net that should land regardless.

### Part A — checkpoint/restore the target's linear-attention state around the verify pass

```diff
--- a/src/cpp/src/speculative_decoding/continuous_batching/pipeline_impl.hpp
+++ b/src/cpp/src/speculative_decoding/continuous_batching/pipeline_impl.hpp
@@ -46,6 +46,12 @@ public:
     size_t get_processed_tokens_per_iteration();
     std::optional<uint64_t> reserve_linear_attention_checkpoints_for_next_step(uint64_t request_id, size_t checkpoint_count);
     void promote_linear_attention_checkpoint_for_sequence(std::optional<uint64_t> seq_id, size_t checkpoint_slot);
     void release_linear_attention_checkpoints_for_sequence(std::optional<uint64_t> seq_id);
+
+    // True when this pipeline's model keeps recurrent (conv/SSM) state, i.e. state that cannot be
+    // trimmed per token and therefore needs checkpoint/promote around speculative verification.
+    bool has_linear_attention_state() const { return m_scheduler && m_scheduler->has_linear_attention_cache(); }
+    // True when the request has finished its prompt and is producing generated tokens.
+    bool is_in_generation_phase(uint64_t request_id) const;
```

```diff
--- a/src/cpp/src/speculative_decoding/continuous_batching/pipeline_impl.cpp
+++ b/src/cpp/src/speculative_decoding/continuous_batching/pipeline_impl.cpp
@@ -545,6 +545,17 @@
 }
 
+bool ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::
+is_in_generation_phase(uint64_t request_id) const {
+    for (const auto& request : m_requests) {
+        if (request->get_request_id() != request_id) {
+            continue;
+        }
+        return request->get_num_processed_tokens() >= request->get_prompt_len();
+    }
+    return false;
+}
+
 std::optional<uint64_t> ContinuousBatchingPipeline::ContinuousBatchingForSpeculativeDecodingImpl::
 reserve_linear_attention_checkpoints_for_next_step(uint64_t request_id, size_t checkpoint_count) {
```

```diff
--- a/src/cpp/src/speculative_decoding/continuous_batching/fast_draft_strategy.hpp
+++ b/src/cpp/src/speculative_decoding/continuous_batching/fast_draft_strategy.hpp
@@ -178,6 +178,12 @@ protected:
     // Mutex protecting access to m_draft_generations, so add_request and step methods can be called from different threads
     std::mutex m_draft_generations_mutex;
     std::map<uint64_t, GenerationHandle> m_draft_generations;
+
+    // Per-round linear-attention checkpoint bookkeeping for the target model (hybrid models only).
+    // request_id -> { sequence id holding the reserved checkpoints, committed generated length
+    //                 before the verify pass, number of candidates submitted for validation }.
+    struct TargetLACheckpoint { std::optional<uint64_t> seq_id; size_t committed_len = 0; size_t candidates = 0; };
+    std::map<uint64_t, TargetLACheckpoint> m_target_la_checkpoints;
+    void release_target_la_checkpoints();
```

```diff
--- a/src/cpp/src/speculative_decoding/continuous_batching/fast_draft_strategy.cpp
+++ b/src/cpp/src/speculative_decoding/continuous_batching/fast_draft_strategy.cpp
@@
 std::pair<ov::genai::SchedulerConfig, ov::genai::SchedulerConfig>
 ContinuousBatchingPipeline::SpeculativeDecodingImpl::init_speculative_models(const ov::genai::ModelDesc& main_model_desc, const ov::genai::ModelDesc& draft_model_desc) {
     auto main_model = main_model_desc.model;
     auto draft_model = draft_model_desc.model;
     OPENVINO_ASSERT(main_model != nullptr, "Main model cannot be null");
     OPENVINO_ASSERT(draft_model != nullptr, "Draft model cannot be null");
 
+    // Recurrent (conv/SSM) state cannot be trimmed per token: rejected candidates would leave the
+    // state permanently ahead of the committed token sequence. The target is handled by
+    // per-token linear-attention checkpoints (see step()), which needs a slightly larger LA pool.
+    // A hybrid *draft* has no such mechanism yet (its state advances one token per infer across
+    // several infers), so reject that pairing instead of silently producing garbage.
+    const bool target_has_linear_attention = utils::get_cache_types(*main_model).has_linear();
+    OPENVINO_ASSERT(!utils::get_cache_types(*draft_model).has_linear(),
+                    "Speculative decoding with a draft model that has linear attention (recurrent) "
+                    "states is not supported: its state cannot be rolled back to the accepted "
+                    "prefix. Use a KV-cache-only draft model, or an MTP/DFlash-style draft head.");
+
     auto main_scheduler_config = main_model_desc.scheduler_config;
@@
     ov::genai::SchedulerConfig main_scheduler_config_updated = main_scheduler_config,
                                draft_scheduler_config = is_draft_scheduler_undefined ? main_scheduler_config : draft_model_desc.scheduler_config;
 
+    if (target_has_linear_attention) {
+        // One committed block + one checkpoint per validated token + slack, cf.
+        // dflash_cb::linear_attention_checkpoint_block_count().
+        auto cfg = main_model_desc.generation_config;
+        // NB: m_main_pipeline does not exist yet here, so use the same default value as
+        // ContinuousBatchingForSpeculativeDecodingImpl::default_num_assistant_tokens (= 5,
+        // pipeline_impl.hpp:16); hoist it to a shared constant when applying.
+        const size_t num_assistant_tokens = cfg.num_assistant_tokens == 0 ? 5 : cfg.num_assistant_tokens;
+        main_scheduler_config_updated.num_linear_attention_blocks =
+            std::max(main_scheduler_config_updated.num_linear_attention_blocks,
+                     num_assistant_tokens + 2);
+    }
+
     if (is_draft_scheduler_undefined) {
@@ void ContinuousBatchingPipeline::SpeculativeDecodingImpl::step() {
     // to generate num_matches statistic
     std::map<int64_t, UpdateRequestResult> update_sequence_info;
+    // Committed generated length per request *before* candidates are appended; needed to derive the
+    // accepted count (and hence the checkpoint slot) after the verify pass.
+    std::map<uint64_t, size_t> committed_len_before;
+    if (m_main_pipeline->has_linear_attention_state()) {
+        for (const auto& r : m_main_pipeline->get_generated_requests()) {
+            committed_len_before[r.first] = r.second.begin()->second.token_ids.size();
+        }
+    }
     // put candidates to model KV cache
     auto draft_generated_requests = m_draft_pipeline->get_generated_requests();
     for (const auto& candidate : m_draft_pipeline->get_generated_requests()) {
         auto update_result = m_main_pipeline->update_request(candidate.first, candidate.second, false);
         update_sequence_info.insert({{candidate.first, update_result}});
+
+        // Hybrid target: make the verify pass write one recurrent-state snapshot per scheduled
+        // token into temporary blocks, so the state at the accepted prefix survives rejection.
+        if (m_main_pipeline->has_linear_attention_state()) {
+            const size_t committed = committed_len_before.count(candidate.first)
+                                         ? committed_len_before.at(candidate.first) : 0;
+            const size_t candidate_len = candidate.second.begin()->second.token_ids.size();
+            const size_t num_candidates = candidate_len > committed ? candidate_len - committed : 0;
+            // Only the generation phase is checkpointable: a prompt-phase chunk would need one
+            // checkpoint block per prompt token. The draft is paused until the target has produced
+            // its first token, so this holds in practice; assert rather than corrupt if it does not.
+            OPENVINO_ASSERT(num_candidates == 0 || m_main_pipeline->is_in_generation_phase(candidate.first),
+                            "Speculative candidates were submitted while the hybrid target model is "
+                            "still processing the prompt; linear attention checkpointing cannot cover "
+                            "a prompt chunk. Disable dynamic_split_fuse for hybrid targets.");
+            if (num_candidates > 0) {
+                TargetLACheckpoint cp;
+                cp.committed_len = committed;
+                cp.candidates = num_candidates;
+                // Verify input is [last committed token, c1..cK] => K+1 scheduled tokens.
+                cp.seq_id = m_main_pipeline->reserve_linear_attention_checkpoints_for_next_step(
+                    candidate.first, num_candidates + 1);
+                m_target_la_checkpoints[candidate.first] = cp;
+            }
+        }
     }
     m_main_pipeline->sync_generated_embeddings();
@@
     const auto main_start = std::chrono::steady_clock::now();
-    m_main_pipeline->step();
+    try {
+        m_main_pipeline->step();
+    } catch (...) {
+        release_target_la_checkpoints();
+        throw;
+    }
     const auto main_end = std::chrono::steady_clock::now();
@@
     auto main_generated_requests = m_main_pipeline->get_generated_requests();
+
+    // Commit the recurrent state that corresponds to exactly the accepted prefix.
+    // produced = accepted candidates + 1 (the target's own token: bonus if all accepted,
+    // correction otherwise)  =>  slot = accepted + 1, because the verify input started with the
+    // last committed token (slot indices are one-based, slot i = state after i scheduled tokens).
+    for (auto& [request_id, cp] : m_target_la_checkpoints) {
+        auto it = main_generated_requests.find(request_id);
+        if (it == main_generated_requests.end() || it->second.empty()) {
+            m_main_pipeline->release_linear_attention_checkpoints_for_sequence(cp.seq_id);
+            continue;
+        }
+        const size_t target_len = it->second.begin()->second.token_ids.size();
+        if (target_len <= cp.committed_len) {
+            m_main_pipeline->release_linear_attention_checkpoints_for_sequence(cp.seq_id);
+            continue;
+        }
+        const size_t produced = target_len - cp.committed_len;
+        const size_t accepted = std::min(cp.candidates, produced - 1);
+        m_main_pipeline->promote_linear_attention_checkpoint_for_sequence(cp.seq_id, accepted + 1);
+    }
+    m_target_la_checkpoints.clear();
+
     for (const auto& checked_sequence : main_generated_requests) {
```

plus, next to `drop_requests()`:

```diff
+void ContinuousBatchingPipeline::SpeculativeDecodingImpl::release_target_la_checkpoints() {
+    for (auto& [_, cp] : m_target_la_checkpoints) {
+        m_main_pipeline->release_linear_attention_checkpoints_for_sequence(cp.seq_id);
+    }
+    m_target_la_checkpoints.clear();
+}
+
 void ContinuousBatchingPipeline::SpeculativeDecodingImpl::drop_requests() {
+    release_target_la_checkpoints();
     m_draft_pipeline->finish_request();
     m_main_pipeline->finish_request();
 }
```

`reserve_...` / `promote_...` / `release_...` are already no-ops when the model has no LA cache
(`scheduler.hpp:305-307`, `pipeline_impl.cpp:549-551`), so pure-KV pairs are unaffected.

### Part B — the draft side

Part A fixes the target. The draft has the same defect at
`pipeline_impl.cpp:449`, and the current checkpoint API cannot express it: the draft advances
**one token per `infer()` across K separate schedules**, while `reserve_temporary_blocks` asserts
the temp table is empty (`block_manager.hpp:957-958`) and is consumed by a single schedule
(`scheduler.hpp:824`). Part A therefore *rejects* a hybrid draft up front (assert above), which is
the honest short-term behaviour and matches the stateful path's existing message.

Full support needs a small, contained extension — worth doing only if a hybrid draft is actually
wanted (for a `qwen35_mtp` strategy it is **not**: follow DFlash and use a hidden-state-fed head
with no CB pipeline of its own):

1. `block_manager.hpp` — add `reserve_additional_temporary_blocks(seq_id, n)` that appends instead
   of asserting empty; make `promote_temporary_block` index the accumulated list.
2. `scheduler.hpp:802-826` — when a checkpoint count is pending and `num_scheduled_tokens == 1`,
   append one temp block per draft decode step (`cache_interval = 1`, `block_indices =
   [committed_or_prev_tmp, next_tmp]`) instead of erasing the reservation after one schedule.
3. `fast_draft_strategy.cpp` — arm the chain before `m_draft_pipeline->multistep()` and, after the
   target's verdict, `promote(slot = accepted)` on the draft (the draft's chain does not include a
   seed token, hence no `+1`), then release the rest.

### Part C — safety net (should land independently, tiny)

CB Eagle3 (`speculative_decoding/continuous_batching/eagle3_strategy.cpp`) has the identical
defect and no guard. Add the same `get_cache_types(*model).has_linear()` assertion there until it
gets checkpointing, so hybrid users get a clear error instead of degenerate text. This mirrors
`stateful/fast_draft_strategy.cpp:55-59`.

## 5. Cost, and whether something cheaper is correct

**Cost of Part A.** Memory: `num_assistant_tokens + 2` linear-attention blocks instead of 1 — one
block holds the full recurrent + conv state of all 24 GDN layers, so for a 9B target with 5
assistant tokens this is ~6-7× the LA cache footprint (identical to what DFlash already pays;
`dflash_strategy_utils.hpp:139-143`). Bandwidth: the verify kernel writes `K+1` state snapshots
instead of overwriting one, i.e. `K` extra state-sized writes per round (the reads are unchanged).
**Restore itself is free**: `promote_temporary_block` swaps block pointers and frees the old block
(`block_manager.hpp:1011-1019`) — no tensor copy, no extra infer. There is no per-round
`state.get_state()/set_state()` copying at all, which is why this is the right mechanism rather
than a `query_state()`-based snapshot (and under PagedAttention the GDN state isn't in
`ov::IVariableState` anyway).

**Cheaper alternatives, and why they lose.**

- *Single pre-verify snapshot + replay.* Copy the committed block once per round (1 state copy),
  and on rejection restore it and re-run the target over just the accepted prefix. Cost: 1 copy
  per round always, plus **one extra target forward pass per rejection**. With realistic
  acceptance rates rejections happen most rounds, so this roughly halves speculative throughput —
  worse than `K` extra state writes. It is, however, ~30 lines and needs no scheduler change, so
  it is a reasonable stopgap if the LA memory increase is unacceptable.
- *Reset + re-prefill on rejection* (what the stateful path does via `needs_reset()`,
  `utils.hpp:227`). Correct but catastrophic: a full prompt re-prefill per rejection.
- *Trimming the recurrent state.* Impossible in principle — GDN state is a multiplicatively
  accumulated matrix with no per-token slot to drop. This is precisely what
  `utils.cpp:571` and `stateful/fast_draft_strategy.cpp:57-59` already assert against.

So: per-token checkpoints (Part A) are both the correct and the fastest option, and they reuse
machinery that is already in the tree and already exercised by DFlash.
