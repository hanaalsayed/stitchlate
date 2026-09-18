import numpy as np

from stitchlate.cleanup import count_isolated, reduce_confetti


def test_removes_a_single_isolated_stitch():
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 1
    assert count_isolated(grid) == 1

    cleaned, changed = reduce_confetti(grid)
    assert cleaned.sum() == 0
    assert changed == 1


def test_leaves_a_solid_block_alone():
    grid = np.zeros((6, 6), dtype=int)
    grid[2:4, 2:4] = 1
    assert count_isolated(grid) == 0

    cleaned, changed = reduce_confetti(grid)
    assert np.array_equal(cleaned, grid)
    assert changed == 0


def test_checkerboard_is_all_isolated():
    """Every stitch differs from all four neighbors, so all of them count."""
    grid = np.indices((6, 6)).sum(0) % 2
    assert count_isolated(grid) == 36


def test_handles_corner_stitches():
    """A corner has only two neighbors, so the edge check has to hold up."""
    grid = np.zeros((4, 4), dtype=int)
    grid[0, 0] = 1
    cleaned, changed = reduce_confetti(grid)
    assert cleaned.sum() == 0
    assert changed == 1


def test_does_not_modify_input():
    grid = np.zeros((5, 5), dtype=int)
    grid[2, 2] = 1
    before = grid.copy()
    reduce_confetti(grid)
    assert np.array_equal(grid, before)


def test_more_passes_remove_more():
    rng = np.random.default_rng(0)
    grid = rng.integers(0, 6, (30, 30))
    one, _ = reduce_confetti(grid, passes=1)
    three, _ = reduce_confetti(grid, passes=3)
    assert count_isolated(three) <= count_isolated(one)