from alignment_selection_dynamics.metrics import summarize


def test_summarize_reports_mean_std_and_n():
    out = summarize([1.0, 2.0, 3.0])
    assert out["mean"] == 2.0
    assert out["n"] == 3
    assert out["std"] > 0.0
