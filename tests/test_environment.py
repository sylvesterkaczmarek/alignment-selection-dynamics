from alignment_selection_dynamics.environment import make_dataset, make_environment_schedule
import pytest
import torch


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


@pytest.mark.parametrize("kwargs", [
    {"shortcut_accuracy": -0.1}, {"cheap_accuracy": 1.1},
    {"shift_fraction": float("nan")}, {"shift_fraction": True},
    {"robust_sigma": -0.5}, {"robust_sigma": float("inf")},
    {"conflict": "false"},
])
def test_invalid_dataset_settings_are_rejected(kwargs):
    with pytest.raises(ValueError):
        make_dataset(16, 7, **kwargs)


@pytest.mark.parametrize("n, seed", [(0, 7), (-1, 7), (True, 7), (16, True), (16, -1), (16, 2**64)])
def test_invalid_dataset_size_or_seed_is_rejected(n, seed):
    with pytest.raises(ValueError):
        make_dataset(n, seed)


def test_probability_endpoints_and_full_width_data_seeds_are_supported():
    x, y = make_dataset(128, 2**64 - 1, shortcut_accuracy=1, cheap_accuracy=0, robust_sigma=0, shift_fraction=1)
    sign = y * 2 - 1
    assert torch.equal(x[:, 0], sign)
    assert torch.all(x[:, 1] * sign < 0)
    assert torch.all(x[:, 2] * sign < 0)


@pytest.mark.parametrize("accuracy,expected_sign", [(0, -1), (1, 1)])
def test_zero_uniform_draw_respects_probability_endpoints(monkeypatch, accuracy, expected_sign):
    # A uniform draw can be exactly zero. Accuracy zero must still always flip.
    monkeypatch.setattr(torch, "rand", lambda n, **kwargs: torch.zeros(n))
    monkeypatch.setattr(torch, "randn", lambda n, **kwargs: torch.zeros(n))
    x, y = make_dataset(16, 7, shortcut_accuracy=accuracy, cheap_accuracy=accuracy)
    expected = expected_sign * (y * 2 - 1)
    assert torch.equal(x[:, 1], expected)
    assert torch.equal(x[:, 2], expected)


@pytest.mark.parametrize("generation", [-1, True, 0.5])
def test_invalid_generation_is_rejected(generation):
    with pytest.raises(ValueError, match="generation"):
        make_environment_schedule("shortcut_challenge", generation)
