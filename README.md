# medrag-das — DAS + RAG for Stochastic Parroting Reduction (reconstruction scaffold)

A reconstructed implementation scaffold for the system in:

> **Dynamic Adapter Swapping and Retrieval-Augmented Generation for Stochastic Parroting
> Reduction in Large Language Models.** Christopher Puglisi, JHU Applied & Computational
> Math, May 2024. Mentor: Zerotti Woods, PhD.

> ### 🛑 Read [`WARNINGS.md`](./WARNINGS.md) first.
> This code produces medical-sounding text, the headline "81.2%" number is easy to misread,
> the original code is lost (this is rebuilt from the paper), and the adapter router in the
> paper was an oracle. All four caveats are explained there. This is a **research
> proof-of-concept, not a medical device.**

---

## What the system is

A lightweight, local pipeline that wraps a quantized **Llama-2-7B** with two parroting-
mitigation mechanisms and routes every question through three units (paper, Fig. 1):

```
                  ┌───────────────── Retrieval Augmentation Unit (RAG) ─────────────────┐
                  │  embed question (BioMedLM) → FAISS IndexFlatL2 → cosine-threshold    │
   question ─────▶│  subset S → return top min(|S|,3) abstracts by L2 distance          │──┐
                  └─────────────────────────────────────────────────────────────────────┘  │
                  ┌──────────── Dynamic Adapter Swapping Unit (DAS) ───────────────────┐    │ context
                  │  pick category → load its QLoRA adapter → fuse with base Llama-2-7B │    │
   question ─────▶│  (21 adapters, one per MedMCQA category; one fused at a time)       │──┐ │
                  └────────────────────────────────────────────────────────────────────┘  │ │
                  ┌──────────────────────── Generation Unit ───────────────────────────┐  │ │
                  │  build prompt (sys + user + retrieved context) → fused LLM →        │◀─┘─┘
   answer ◀───────│  JSON {cop, cop index, why correct}  (parsed best-effort)           │
                  └─────────────────────────────────────────────────────────────────────┘
```

- **Base model:** `meta-llama/Llama-2-7b`, 8-bit quantized, run via **Apple MLX**.
- **RAG KB:** >100k PubMed OpenAccess abstracts (TF `scientific_papers`), embedded with
  `stanford-crfm/BioMedLM`, stored in a FAISS `IndexFlatL2`.
- **DAS:** 21 QLoRA adapters, each fine-tuned on one MedMCQA category subset (≤10k samples,
  early-stop on train-loss plateau, 4 LoRA layers, batch size 2).
- **Selection:** oracle (ground-truth category) in the paper; a classifier stub is provided
  for real deployment (see `WARNINGS.md` §4).

## Reported results (paper, Table 1 — for context, not a target to overclaim)

| System | MedMCQA | Med-HALT FAKE | Med-HALT NOTA |
|---|---|---|---|
| Llama-2-7B (baseline) | 25.01 | 58.72 | 69.49 |
| Llama-2-7B + DAS | 25.21 | 63.20 | 71.55 |
| Llama-2-7B + RAG | 25.10 | 60.78 | 79.90 |
| **Llama-2-7B + DAS + RAG** | **25.18** | 62.80 | **81.20** |
| Llama-2-70B (ref) | NA | 97.26 | 77.53 |
| GPT-3.5 (ref) | 49.10 | 71.64 | 27.64 |

**MedMCQA ≈ 25% = chance.** The gains are on NOTA/FAKE (abstention-friendly tests). See
`WARNINGS.md` §3.

## Layout

```
configs/            default.yaml (all knobs) · categories.yaml (the 21 adapters)
src/medrag_das/
  config.py         typed config loader
  backends/         base.py (interface) · mlx_backend.py (primary) · torch_backend.py (stub)
  retrieval/        embedder.py · build_kb.py · search.py   (RAG unit)
  adapters/         train_adapter.py · train_all.py · registry.py · selector.py  (DAS unit)
  generation/       prompts.py (Appendix Table 4) · generate.py                 (Gen unit)
  pipeline.py       end-to-end orchestration
  parsing.py        robust {cop, cop index, why correct} extraction
  cli.py            `medrag` CLI: build-kb | train | eval | ask
eval/               datasets.py · run_eval.py (4 methods × 3 datasets) · metrics.py
scripts/            download_data.sh · smoke_test.py
tests/              parsing / search / prompts
```

## Quickstart

```bash
# 0. Environment (Apple Silicon)
python -m venv .venv && source .venv/bin/activate
pip install -e .                      # see pyproject.toml / requirements.txt
cp .env.template .env                 # fill in paths + HF token

# 1. Build the PubMed knowledge base (slow; see WARNINGS §7)
medrag build-kb --limit 100000

# 2. Train the 21 category adapters (QLoRA via MLX)
medrag train --all                    # or: medrag train --category pharmacology

# 3. Evaluate the 4-method × 3-dataset matrix
medrag eval --methods baseline das rag das_rag --datasets medmcqa nota fake

# 4. Ask one question interactively
medrag ask --question "..." --options "A ... B ... C ... D ..." --category pharmacology
```

All four commands are **fully implemented** (no stubs left). What they still need to actually
run is **external**: a Hugging Face token with Llama-2 access, the datasets, and on-device
compute — none of which ship in the repo. The code is written against the `mlx-lm` /
`transformers` / `datasets` / `faiss` APIs as of early 2026; **pin your versions** and treat
your first successful end-to-end run as the real validation (see `WARNINGS.md §1`).

## Status / implementation checklist

**Implemented + tested offline (26 tests, no GPU/weights/downloads):**
- [x] Project structure, config, categories, prompts (verbatim from Appendix Table 4)
- [x] Robust output parser; semantic-search math (cosine-threshold → top-3 L2)
- [x] Oracle **and** classifier adapter routers; adapter registry/swap
- [x] Pipeline orchestration for all 4 methods; metrics; dataset loaders (fixture path)
- [x] MLX backend (load/generate/fuse/QLoRA-train) + Torch/CUDA backend
- [x] BioMedLM embedder + FAISS KB build + PubMed streaming ingest
- [x] Fixtures + GitHub Actions CI

**Needs a real run to validate (external deps, can't be done offline):**
- [ ] First end-to-end run on Apple Silicon with Llama-2 weights (pin `mlx-lm` version)
- [ ] Verify Med-HALT HF ids/fields (marked `VERIFY` in `evaluation/datasets.py`)
- [ ] Reproduce Table 1 and fill in `RESULTS.md`
