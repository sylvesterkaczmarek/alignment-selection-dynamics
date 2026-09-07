import itertools

import pytest

from alignment_selection_dynamics.seeding import derive_data_seed, validate_seed


@pytest.mark.parametrize("seed", [-1, 2**32, True, 1.5, "7"])
def test_invalid_root_seeds_are_rejected(seed):
    with pytest.raises(ValueError, match="seed"):
        validate_seed(seed)


@pytest.mark.parametrize("seed", [0, 2**32 - 1])
def test_root_seed_boundaries_are_supported(seed):
    validate_seed(seed)
    assert 0 <= derive_data_seed(seed, 0, 0, 0) < 2**64


def test_old_population_and_generation_stride_collisions_are_separated():
    # Both pairs shared a seed under seed*100000 + generation*1000 + agent*10.
    assert derive_data_seed(1, 0, 100, 0) != derive_data_seed(1, 1, 0, 0)
    assert derive_data_seed(1, 100, 0, 0) != derive_data_seed(2, 0, 0, 0)


def test_data_streams_are_repeatable_and_distinct_across_coordinates():
    coordinates = list(itertools.product([1, 2], [0, 1, 100], [0, 1, 100], range(3)))
    first = [derive_data_seed(*coordinate) for coordinate in coordinates]
    assert first == [derive_data_seed(*coordinate) for coordinate in coordinates]
    assert len(first) == len(set(first))


@pytest.mark.parametrize("coordinate", [(1, -1, 0, 0), (1, 0, True, 0), (1, 0, 0, 3)])
def test_invalid_stream_coordinates_are_rejected(coordinate):
    with pytest.raises(ValueError):
        derive_data_seed(*coordinate)
