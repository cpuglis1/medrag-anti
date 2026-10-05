"""Dataset-loader tests — exercise the fixture path only (no HF download, no weights)."""
from medrag_das.config import load_config

from medrag_das.evaluation.datasets import (
    FIXTURE_PATHS,
    load_medhalt_fake,
    load_medhalt_nota,
    load_medmcqa_test,
)


def test_fixtures_exist():
    for name, path in FIXTURE_PATHS.items():
        assert path.exists(), f"missing fixture for {name}: {path}"


def test_medmcqa_fixture_normalizes():
    items = list(load_medmcqa_test({}, fixture=FIXTURE_PATHS["medmcqa"]))
    assert len(items) == 6
    first = items[0]
    assert first.gold_index == 1                       # 1-based
    assert len(first.options) == 4
    assert first.correct_is_abstain is False
    # Every gold_label must be a real adapter slug (so the oracle router can use it).
    valid = set(load_config().category_slugs)
    assert all(it.gold_label in valid for it in items)


def test_nota_fixture_points_at_none_of_the_above():
    items = list(load_medhalt_nota(fixture=FIXTURE_PATHS["nota"]))
    assert len(items) == 4
    for it in items:
        assert it.correct_is_abstain is False          # NOTA is an explicit option
        assert it.gold_index is not None
        # gold option text should be the NOTA entry
        assert it.options[it.gold_index - 1].lower().startswith("none of the above")


def test_fake_fixture_is_unanswerable():
    items = list(load_medhalt_fake(fixture=FIXTURE_PATHS["fake"]))
    assert len(items) == 4
    for it in items:
        assert it.correct_is_abstain is True
        assert it.gold_index is None


def test_limit_is_respected():
    items = list(load_medmcqa_test({}, limit=2, fixture=FIXTURE_PATHS["medmcqa"]))
    assert len(items) == 2
