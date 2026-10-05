from medrag_das.evaluation.metrics import score


def test_accuracy_balanced_and_rates():
    preds = [
        {"gold_index": 1, "pred_index": 1, "parse_failed": False, "abstained": False, "gold_label": "a"},
        {"gold_index": 2, "pred_index": 2, "parse_failed": False, "abstained": False, "gold_label": "b"},
        {"gold_index": 3, "pred_index": 4, "parse_failed": False, "abstained": False, "gold_label": "a"},
        {"gold_index": 1, "pred_index": None, "parse_failed": True, "abstained": True, "gold_label": "b"},
    ]
    r = score(preds)
    assert r.n == 4
    assert r.n_scored == 3                       # parse failure excluded from scoring
    assert abs(r.accuracy - 2 / 3) < 1e-9
    assert abs(r.parse_fail_rate - 0.25) < 1e-9
    assert abs(r.abstain_rate - 0.25) < 1e-9
    # class a: 1/2 recall, class b: 1/1 recall -> balanced 0.75
    assert abs(r.balanced_accuracy - 0.75) < 1e-9


def test_empty_is_safe():
    r = score([])
    assert r.n == 0 and r.accuracy == 0.0 and r.balanced_accuracy == 0.0
