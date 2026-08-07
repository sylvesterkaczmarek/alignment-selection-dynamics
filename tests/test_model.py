from alignment_selection_dynamics.model import AgentTraits, SelectionAgent, logistic


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
