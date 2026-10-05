from medrag_das.parsing import parse_answer


def test_clean_json():
    raw = '{"cop": "Aspirin", "cop index": 2, "why correct": "it inhibits COX"}'
    a = parse_answer(raw)
    assert a.cop == "Aspirin"
    assert a.cop_index == 2
    assert not a.parse_failed


def test_single_quoted_json_repair():
    raw = "Sure! {'cop': 'B', 'cop index': '3', 'why correct': 'because'}"
    a = parse_answer(raw)
    assert a.cop_index == 3
    assert not a.parse_failed


def test_prose_fallback_option_number():
    raw = "After reasoning, the correct option is 4 since the others are wrong."
    a = parse_answer(raw)
    assert a.cop_index == 4


def test_abstention_detected():
    raw = "I must respectfully point out this is fictional. I do not know."
    a = parse_answer(raw)
    assert a.abstained is True


def test_nota_phrase_is_abstention():
    raw = "The answer is None of the Above."
    a = parse_answer(raw)
    assert a.abstained is True


def test_total_garbage_is_parse_failure():
    a = parse_answer("purple monkey dishwasher")
    assert a.parse_failed is True
    assert a.raw == "purple monkey dishwasher"
