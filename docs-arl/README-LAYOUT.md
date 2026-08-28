# Tree layout

Reorganised 2026-08-03. Previously everything sat flat at the root (72 entries, 41 of them
model directories) and every path was resolved **relative to the CWD** — which worked only
because `ornith-run.sh` happened to be invoked from this directory.

```
run/        model runners. ov_boot.py is the tree anchor: it exports ROOT, MODELS and
            model(name) -> absolute path. Every OV runner imports it, so this is the one
            place that knows the layout. Entry point: run/ornith-run.sh (works from any CWD).
models/     all model weights and drafts, referenced by bare name through ov_boot.model()
tools/      one-shot utilities (exporters, requantisers, transplant helpers)
docs/       PERF-FINDINGS.md, KAT-QUANT-SWEEPS.md, QUANT-STATUS.md, ...
evals/      aquarium (single-prompt, historical) and suite (8 prompts, greedy, blind scoring)
bin/        compiled helpers only — dropcaches is setuid root, leave it alone
genai-src/  patched OpenVINO GenAI. build/ is the ABI-pinned default; build-xg/ adds
            ENABLE_XGRAMMAR=ON for structured output (select with OV_GENAI_BUILD=)
kaggle-quant/ quantisation sweep tooling; runs/ holds spent kernel-staging dirs
arena/ attic/ gemma4-ov/ ornith_transplant/ benchmax_eval/   historical, not maintained
```

Adding a model: drop it in `models/`, add an entry to `MODELS` in `run/ov_serve.py`, and a
case to `run/ornith-run.sh` (model alias, `ENGINE=ov`, and a `MEM=` line — that case has no
default, so a missing entry fails with `MEM: unbound variable`).
