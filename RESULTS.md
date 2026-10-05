# Results

> Fill the "Reproduced" columns after running `medrag eval`. Keep the paper's numbers next to
> yours so deviations are visible. **Read `WARNINGS.md §3` before interpreting any of these** —
> MedMCQA ≈ 25% is chance, and the NOTA/FAKE gains track abstention, not medical knowledge.

## Paper, Table 1 (reference)

| System | MedMCQA | Med-HALT FAKE | Med-HALT NOTA |
|---|---|---|---|
| Llama-2-7B (baseline) | 25.01 | 58.72 | 69.49 |
| Llama-2-7B + DAS | 25.21 | 63.20 | 71.55 |
| Llama-2-7B + RAG | 25.10 | 60.78 | 79.90 |
| **Llama-2-7B + DAS + RAG** | **25.18** | 62.80 | **81.20** |
| Llama-2-70B (ref) | NA | 97.26 | 77.53 |
| GPT-3.5 (ref) | 49.10 | 71.64 | 27.64 |

## Reproduced (this scaffold)

Report accuracy **with** the abstain-rate and parse-fail-rate the harness emits — a high
NOTA/FAKE accuracy next to a high abstain-rate is the expected (and honest) pattern.

| System | MedMCQA acc | NOTA acc | FAKE acc | abstain-rate (NOTA/FAKE) | parse-fail-rate |
|---|---|---|---|---|---|
| Llama-2-7B (baseline) | | | | | |
| Llama-2-7B + DAS | | | | | |
| Llama-2-7B + RAG | | | | | |
| Llama-2-7B + DAS + RAG | | | | | |

**Routing used:** `oracle` / `classifier` (state which — they are not comparable; see
`WARNINGS.md §4`).

## Environment & deviations (fill in)

- Hardware: (e.g. Apple M2 Pro 16 GB) · backend: mlx / torch
- Versions: `mlx-lm`, `transformers`, `datasets`, `faiss` (pin them)
- Settings chosen for `PAPER UNSPECIFIED` knobs: cosine `t` = …, LoRA rank/alpha = …, LR = …
- KB: N abstracts, index size, avg retrieval latency
- Adapters: iters-to-plateau per category, total adapter storage
