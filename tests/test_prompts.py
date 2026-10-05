from medrag_das.adapters.selector import OracleSelector
from medrag_das.generation.prompts import build_prompt


def test_prompt_includes_system_and_instruction_markers():
    p = build_prompt("What is X?", ["a", "b", "c", "d"])
    assert p.startswith("<s>[INST] <<SYS>>")
    assert "[/INST]" in p
    assert "1) a" in p and "4) d" in p
    assert "JSON" in p


def test_context_only_present_when_supplied():
    without = build_prompt("Q?", ["a", "b"])
    assert "abstracts context" not in without
    with_ctx = build_prompt("Q?", ["a", "b"], context=["abstract one"])
    assert "abstracts context" in with_ctx
    assert "abstract one" in with_ctx


def test_oracle_selector_trusts_gold_label():
    sel = OracleSelector(["pharmacology", "surgery", "unknown"])
    assert sel.select("q", gold_category_slug="surgery") == "surgery"
    # unknown-valid label falls back to 'unknown'
    assert sel.select("q", gold_category_slug="cardiology") == "unknown"


def test_oracle_selector_random_for_fake():
    sel = OracleSelector(["pharmacology", "surgery", "unknown"], fake_strategy="random", seed=1)
    choice = sel.select("fabricated scenario", gold_category_slug=None)
    assert choice in {"pharmacology", "surgery", "unknown"}
