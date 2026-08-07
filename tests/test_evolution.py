from alignment_selection_dynamics.evolution import EvolutionConfig, run_evolution


def small_config() -> EvolutionConfig:
    return EvolutionConfig(
        population_size=6,
        generations=4,
        parent_pool=3,
        elites=1,
        life_epochs=2,
        train_examples=96,
        eval_examples=128,
    )


def test_evolution_is_repeatable_for_same_seed():
    cfg = small_config()
    run_a = run_evolution("capability_positive", 11, cfg)
    run_b = run_evolution("capability_positive", 11, cfg)
    assert run_a == run_b


def test_history_has_expected_metrics():
    run = run_evolution("safety_only", 13, small_config())
    row = run["history"][-1]
    for key in [
        "fitness",
        "capability",
        "alignment_score",
        "verifier_strength",
        "bypass_strength",
        "selection_differential",
    ]:
        assert key in row
