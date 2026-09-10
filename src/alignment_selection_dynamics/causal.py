"""Paired, checkpointed experiments; the historical evolution runner is unchanged."""
from __future__ import annotations

import hashlib
import json
import random
from copy import deepcopy
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import torch

from .environment import EnvironmentSpec, make_dataset
from .evolution import EvolutionConfig, _binary_accuracy, _require_finite, _train_agent
from .model import SelectionAgent
from .seeding import validate_seed

PROTOCOL = "paired-v1"
STREAMS = {name: i for i, name in enumerate((
    "train", "fitness", "report", "conflict", "initial_weights", "initial_traits",
    "parents", "trait_mutation", "weight_mutation",
))}


def protocol_seed(seed: int, generation: int, slot: int, stream: str) -> int:
    validate_seed(seed)
    if any(type(v) is not int or v < 0 for v in (generation, slot)):
        raise ValueError("generation and slot must be nonnegative integers")
    if stream not in STREAMS:
        raise ValueError(f"Unknown paired-protocol stream: {stream}")
    sequence = np.random.SeedSequence([0x41534431, seed, generation, slot, STREAMS[stream]])
    return int(sequence.generate_state(1, dtype=np.uint64)[0])


@dataclass(frozen=True)
class Controls:
    selection: str = "fitness"
    mutate_traits: bool = True
    freeze: str = "none"

    def __post_init__(self) -> None:
        if self.selection not in ("fitness", "random"):
            raise ValueError("selection must be fitness or random")
        if type(self.mutate_traits) is not bool:
            raise ValueError("mutate_traits must be boolean")
        if self.freeze not in ("none", "robust", "all"):
            raise ValueError("freeze must be none, robust or all")


@dataclass
class Checkpoint:
    """Population before learning at next_generation, after preceding reproduction.

    Adam is recreated by _train_agent each generation; no optimizer crosses this
    boundary. Initialisation is finished, and all remaining random states are here.
    """
    seed: int
    next_generation: int
    population: list[SelectionAgent]
    births: list[dict]
    parent_state: tuple
    trait_state: tuple
    weight_state: torch.Tensor
    config: dict
    controls: dict


def initial_checkpoint(seed: int, cfg: EvolutionConfig) -> Checkpoint:
    validate_seed(seed)
    if not isinstance(cfg, EvolutionConfig):
        raise TypeError("cfg must be an EvolutionConfig")
    traits = np.random.default_rng(protocol_seed(seed, 0, 0, "initial_traits"))
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(protocol_seed(seed, 0, 0, "initial_weights"))
        population = [SelectionAgent(float(traits.normal(-0.3, 0.35)),
                                     float(traits.normal(-2.5, 0.25)))
                      for _ in range(cfg.population_size)]
    return Checkpoint(
        seed, 0, population,
        [{"id": f"{seed}:0:{i}", "parent_id": None, "mutation": None}
         for i in range(cfg.population_size)],
        random.Random(protocol_seed(seed, 0, 0, "parents")).getstate(),
        random.Random(protocol_seed(seed, 0, 0, "trait_mutation")).getstate(),
        torch.Generator().manual_seed(protocol_seed(seed, 0, 0, "weight_mutation")).get_state(),
        asdict(cfg), asdict(Controls()),
    )


def _payload(state: Checkpoint) -> dict:
    agents = []
    for agent in state.population:
        for parameter in agent.parameters():
            _require_finite(parameter, "checkpoint parameter")
        agents.append({
            "weights": {k: v.detach().clone() for k, v in agent.state_dict().items()},
            "traits": asdict(agent.traits), "training": agent.training,
            "requires_grad": {k: v.requires_grad for k, v in agent.named_parameters()},
        })
    return {
        "protocol": PROTOCOL, "seed": state.seed, "next_generation": state.next_generation,
        "agents": agents, "births": deepcopy(state.births),
        "parent_state": state.parent_state, "trait_state": state.trait_state,
        "weight_state": state.weight_state.clone(), "config": state.config,
        "controls": state.controls,
    }


def checkpoint_fingerprint(state: Checkpoint) -> str:
    """Hash ordered state, independently of Torch archive filenames."""
    payload = _payload(state)
    digest = hashlib.sha256()
    digest.update(payload.pop("weight_state").numpy().tobytes())
    for agent in payload["agents"]:
        weights = agent.pop("weights")
        agent["weight_spec"] = [(k, str(v.dtype), list(v.shape)) for k, v in weights.items()]
        for tensor in weights.values():
            digest.update(tensor.contiguous().numpy().tobytes())
    digest.update(json.dumps(payload, sort_keys=True, allow_nan=False).encode())
    return digest.hexdigest()


def save_checkpoint(path: str | Path, state: Checkpoint) -> None:
    payload = _payload(state)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("xb") as handle:
        torch.save(payload, handle)


def load_checkpoint(path: str | Path) -> Checkpoint:
    data = torch.load(path, map_location="cpu", weights_only=True)
    if data["protocol"] != PROTOCOL:
        raise ValueError("Incompatible checkpoint protocol")
    validate_seed(data["seed"])
    cfg = EvolutionConfig(**data["config"])
    Controls(**data["controls"])
    if type(data["next_generation"]) is not int or data["next_generation"] < 0:
        raise ValueError("Invalid checkpoint boundary")
    population = []
    with torch.random.fork_rng(devices=[]):
        for item in data["agents"]:
            agent = SelectionAgent(**item["traits"])
            agent.to(dtype=next(iter(item["weights"].values())).dtype)
            agent.load_state_dict(item["weights"], strict=True)
            agent.train(item["training"])
            for name, parameter in agent.named_parameters():
                _require_finite(parameter, "checkpoint parameter")
                parameter.requires_grad_(item["requires_grad"][name])
            population.append(agent)
    if len(population) != cfg.population_size or len(data["births"]) != len(population):
        raise ValueError("Checkpoint population size mismatch")
    return Checkpoint(
        data["seed"], data["next_generation"], population, data["births"],
        data["parent_state"], data["trait_state"], data["weight_state"],
        data["config"], data["controls"],
    )


def _reproduce(state: Checkpoint, scores: list[dict], cfg: EvolutionConfig,
               controls: Controls) -> tuple[list[dict], dict]:
    parents_rng, traits_rng = random.Random(), random.Random()
    parents_rng.setstate(state.parent_state)
    traits_rng.setstate(state.trait_state)
    weights_rng = torch.Generator().set_state(state.weight_state)
    if controls.selection == "random":
        order = parents_rng.sample(range(len(scores)), len(scores))
    else:
        order = sorted(range(len(scores)), key=lambda i: -scores[i]["fitness"])
    eligible = order[:cfg.parent_pool]
    chosen = eligible[:cfg.elites] + [parents_rng.choice(eligible)
                                    for _ in range(cfg.population_size - cfg.elites)]
    counts = [chosen.count(i) for i in range(cfg.population_size)]
    records = [dict(state.births[i], **score, eligible=i in eligible,
                    elite_copies=int(i in eligible[:cfg.elites]), offspring_count=counts[i])
               for i, score in enumerate(scores)]
    old_mean = float(np.mean([a.traits.verifier_strength for a in state.population]))
    parent_mean = float(np.mean([state.population[i].traits.verifier_strength for i in chosen]))
    pool_mean = float(np.mean([state.population[i].traits.verifier_strength for i in eligible]))
    offspring, births = [], []
    for slot, parent in enumerate(chosen):
        child = state.population[parent].clone()
        mutation = {"verifier_logit": 0.0, "bypass_logit": 0.0, "weights": {}}
        if slot >= cfg.elites:
            # Draw even for frozen controls, preserving matched mutation streams.
            for name in ("verifier_logit", "bypass_logit"):
                delta = traits_rng.gauss(0.0, cfg.trait_mutation_std)
                if controls.mutate_traits:
                    setattr(child, name, getattr(child, name) + delta)
                    mutation[name] = delta
            with torch.no_grad():
                for name, parameter in child.named_parameters():
                    delta = cfg.weight_mutation_std * torch.randn(
                        parameter.shape, dtype=parameter.dtype, generator=weights_rng)
                    frozen = controls.freeze == "all" or (
                        controls.freeze == "robust" and name.startswith("robust_head."))
                    if frozen:
                        delta.zero_()
                    parameter.add_(delta)
                    _require_finite(parameter, "mutated model parameter")
                    mutation["weights"][name] = delta.tolist()
        child.traits  # Validate finite inherited logits, including extreme mutations.
        births.append({"id": f"{state.seed}:{state.next_generation + 1}:{slot}",
                       "parent_id": state.births[parent]["id"], "mutation": mutation})
        offspring.append(child)
    newborn_mean = float(np.mean([a.traits.verifier_strength for a in offspring]))
    accounting = {"selection_differential": pool_mean - old_mean,
                  "realized_selection_differential": parent_mean - old_mean,
                  "mutation_change": newborn_mean - parent_mean,
                  "newborn_verifier_change": newborn_mean - old_mean}
    state.population, state.births = offspring, births
    state.parent_state, state.trait_state = parents_rng.getstate(), traits_rng.getstate()
    state.weight_state = weights_rng.get_state()
    return records, accounting


def advance(checkpoint: Checkpoint, environments: Sequence[EnvironmentSpec],
            cfg: EvolutionConfig, controls: Controls = Controls(), *,
            diagnostics: bool = True, report_examples: int | None = None
            ) -> tuple[Checkpoint, list[dict], list[dict]]:
    """Run a bounded segment without modifying the supplied checkpoint.

    Selection sees one shared fresh fitness batch. Reporting and conflict examples
    have separate streams and never enter reproduction. Population slots are matched
    across branches; descendants are not independent experimental units.
    """
    if not isinstance(cfg, EvolutionConfig) or not isinstance(controls, Controls):
        raise TypeError("Expected EvolutionConfig and Controls")
    if cfg.population_size != len(checkpoint.population):
        raise ValueError("Cannot change population size at a checkpoint")
    if len(checkpoint.births) != cfg.population_size:
        raise ValueError("Checkpoint ancestry size mismatch")
    if checkpoint.next_generation + len(environments) > cfg.generations:
        raise ValueError("Segment exceeds configured generation horizon")
    if not all(isinstance(env, EnvironmentSpec) for env in environments):
        raise TypeError("Every generation requires an EnvironmentSpec")
    if type(diagnostics) is not bool:
        raise ValueError("diagnostics must be boolean")
    report_examples = cfg.eval_examples if report_examples is None else report_examples
    if type(report_examples) is not int or report_examples < 1:
        raise ValueError("report_examples must be a positive integer")
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    state = deepcopy(checkpoint)
    history, individuals = [], []
    for env in environments:
        generation = state.next_generation
        seeds = {name: protocol_seed(state.seed, generation, 0, name)
                 for name in ("fitness", "report", "conflict")}
        fx, fy = make_dataset(cfg.eval_examples, seeds["fitness"], shortcut_accuracy=0.98,
                             cheap_accuracy=env.cheap_eval_accuracy,
                             shift_fraction=env.eval_shift_fraction)
        if diagnostics:
            rx, ry = make_dataset(report_examples, seeds["report"], shortcut_accuracy=0.98,
                                 cheap_accuracy=env.cheap_eval_accuracy,
                                 shift_fraction=env.eval_shift_fraction)
            px, py = make_dataset(report_examples, seeds["conflict"], conflict=True)
        scores = []
        for slot, agent in enumerate(state.population):
            for name, parameter in agent.named_parameters():
                frozen = controls.freeze == "all" or (
                    controls.freeze == "robust" and name.startswith("robust_head."))
                parameter.requires_grad_(not frozen)
            train_seed = protocol_seed(state.seed, generation, slot, "train")
            if controls.freeze != "all":
                _train_agent(agent, env, train_seed, cfg)
            with torch.no_grad():
                capability = _binary_accuracy(agent(fx)["logit"], fy)
                traits = agent.traits
                cost = env.verifier_cost * traits.verifier_strength
                score = {"fitness_capability": capability, "fitness": capability - cost,
                         "verification_cost": cost, "verifier_strength": traits.verifier_strength,
                         "bypass_strength": traits.bypass_strength,
                         "verifier_logit": agent.verifier_logit, "bypass_logit": agent.bypass_logit}
                if diagnostics:
                    score["capability"] = _binary_accuracy(agent(rx)["logit"], ry)
                    score["alignment_score"] = _binary_accuracy(agent(px)["logit"], py)
            scores.append(score)
        records, accounting = _reproduce(state, scores, cfg, controls)
        for slot, record in enumerate(records):
            record.update(generation=generation,
                          train_seed=protocol_seed(state.seed, generation, slot, "train"))
        individuals.extend(records)
        row = {key: float(np.mean([score[key] for score in scores])) for key in scores[0]}
        row.update(generation=generation, environment=asdict(env), **accounting,
                   best_fitness=max(score["fitness"] for score in scores), data_seeds=seeds)
        if diagnostics:
            row["feature_policies"] = {
                name: {"capability": _binary_accuracy(rx[:, column], ry),
                       "alignment_score": _binary_accuracy(px[:, column], py)}
                for name, column in (("robust", 0), ("shortcut", 1), ("cheap", 2))
            }
        history.append(row)
        state.next_generation += 1
    state.config, state.controls = asdict(cfg), asdict(controls)
    return state, history, individuals
