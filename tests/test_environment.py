from alignment_selection_dynamics.environment import make_dataset, make_environment_schedule


def test_conflict_dataset_reverses_shortcut_and_cheap_signals():
    x, y = make_dataset(2048, 7, conflict=True)
    sign = y * 2 - 1
    assert float((x[:, 1] * sign < 0).float().mean()) > 0.98
    assert float((x[:, 2] * sign < 0).float().mean()) > 0.98
    assert float((x[:, 0] * sign > 0).float().mean()) > 0.85


def test_regime_schedule_changes_shortcut_phase():
    assert make_environment_schedule("shortcut_challenge", 9).label == "pre-challenge"
    assert make_environment_schedule("shortcut_challenge", 10).label == "cheap-shortcut"
    assert make_environment_schedule("shortcut_challenge", 20).label == "recovery"
