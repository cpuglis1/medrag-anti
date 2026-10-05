#!/usr/bin/env bash
# Pointers for fetching the assets this project needs. NOTHING is downloaded automatically —
# you must accept each source's license yourself (see WARNINGS.md §6). Fill in HF_TOKEN first.
set -euo pipefail

echo "This script only prints instructions; it does not download anything."
cat <<'EOF'

1) Llama-2-7B base (gated):
   - Accept the license at https://huggingface.co/meta-llama/Llama-2-7b
   - `huggingface-cli login`  (uses HF_TOKEN)
   - mlx: convert/quantize with `python -m mlx_lm.convert --hf-path meta-llama/Llama-2-7b -q`

2) BioMedLM embedder:
   - https://huggingface.co/stanford-crfm/BioMedLM  (check license on the model card)

3) MedMCQA (training + 4k test):
   - https://huggingface.co/datasets/openlifescienceai/medmcqa

4) Med-HALT (NOTA + FAKE reasoning tests):
   - Pal et al. 2023, arXiv:2307.15343 — follow the dataset release terms.

5) PubMed abstracts for the KB (>100k):
   - TF scientific_papers/pubmed  OR an equivalent HF mirror. Respect PubMed/NCBI terms.
   - Keep the KB independent of MedMCQA/Med-HALT content to avoid leakage (paper §4.1).

Do NOT commit any of these into the repo (.gitignore already blocks weights/indexes/data).
EOF
