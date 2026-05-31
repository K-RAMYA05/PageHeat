from pageheat_app.metrics import exact_match, mcq_exact_match, token_f1


def test_exact_match_normalizes_case_and_space():
    assert exact_match(" Hello   World ", "hello world") == 1.0


def test_token_f1_partial_overlap():
    score = token_f1("a b c", "a c d")
    assert 0.0 < score < 1.0


def test_mcq_exact_match_extracts_letter():
    assert mcq_exact_match("The correct answer is (C).", "C") == 1.0
