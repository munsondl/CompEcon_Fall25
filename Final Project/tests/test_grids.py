from src import grids

def test_integer_grid_basic():
    g = grids.integer_grid(0, 10)
    assert list(g) == list(range(11))

def test_integer_grid_negative():
    g = grids.integer_grid(-2, 2)
    assert list(g) == [-2, -1, 0, 1, 2]

def test_uniform_grid_simple():
    g = grids.uniform_grid(0, 1, 5)
    assert len(g) == 5
    assert g[0] == 0.0
    assert g[-1] == 1.0
