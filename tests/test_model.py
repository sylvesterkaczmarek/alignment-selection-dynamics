from alignment_selection_dynamics.model import AgentTraits, SelectionAgent, logistic
import pytest
import torch


def test_logistic_and_traits_are_bounded():
    assert 0.0 < logistic(-100.0) < 1.0
    traits = AgentTraits(1.0, -1.0)
    assert 0.0 < traits.verifier_strength < 1.0
    assert 0.0 < traits.bypass_strength < 1.0


def test_clone_preserves_traits_and_weights():
    agent = SelectionAgent(0.4, -1.2)
    clone = agent.clone()
    assert clone.verifier_logit == agent.verifier_logit
    assert clone.bypass_logit == agent.bypass_logit
    for p1, p2 in zip(agent.parameters(), clone.parameters()):
        assert p1 is not p2
        assert (p1 == p2).all()


def test_extreme_finite_logits_are_stable():
    assert logistic(-1000.0) == 0.0
    assert logistic(1000.0) == 1.0
    assert logistic(0.0) == 0.5
    assert logistic(-2.0) == pytest.approx(1.0 - logistic(2.0))


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -float("inf"), True, "1"])
def test_invalid_trait_cannot_reach_the_model(value):
    with pytest.raises(ValueError, match="finite numbers"):
        SelectionAgent(verifier_logit=value)
    with pytest.raises(ValueError, match="finite numbers"):
        AgentTraits(0.0, value)


def test_clone_preserves_predictions_precision_mode_and_rng():
    agent = SelectionAgent(0.4, -1.2).double().eval()
    agent.robust_head.weight.requires_grad_(False)
    x = torch.tensor([[0.3, -0.2, 0.1, 1.5, -0.7]], dtype=torch.float64)
    rng = torch.random.get_rng_state().clone()
    child = agent.clone()
    assert torch.equal(rng, torch.random.get_rng_state())
    assert child.training is False
    assert child.robust_head.weight.requires_grad is False
    for key, prediction in agent(x).items():
        torch.testing.assert_close(child(x)[key], prediction, rtol=0, atol=0)
    with torch.no_grad():
        child.shortcut_head.weight.add_(1)
    assert not torch.equal(child.shortcut_head.weight, agent.shortcut_head.weight)
