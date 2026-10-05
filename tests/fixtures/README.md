# Test fixtures — tiny, synthetic, safe to commit

These are **handwritten, fabricated** rows in the project's **normalized EvalItem schema**,
used so the eval harness and dataset loaders can be exercised offline (CI, smoke tests)
without downloading MedMCQA / Med-HALT. They are NOT real exam questions and carry no
dataset license.

Normalized schema (one JSON object per line):

```json
{
  "question": "string",
  "options": ["opt1", "opt2", "..."],
  "gold_index": 2,            // 1-based index of the correct option; null if correct==abstain
  "gold_label": "pharmacology", // adapter slug for DAS oracle (MedMCQA only); else null
  "correct_is_abstain": false // true for FAKE-style unanswerable items
}
```

Conventions mirror the real datasets (paper §4.1):
- `medmcqa_sample.jsonl` — 4 options, one correct, carries a category slug.
- `nota_sample.jsonl` — "None of the above" is an explicit option and is the correct one
  (`gold_index` points to it; `correct_is_abstain: false`).
- `fake_sample.jsonl` — fabricated scenario, no determinable answer
  (`gold_index: null`, `correct_is_abstain: true`).
