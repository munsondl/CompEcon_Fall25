# tests/test_payoffs.py

from src.payoffs import profit
from src.config import load_params

def test_profit_basic():
    params = load_params()

    k = 1
    # pick any z (e.g., mean spot rate), because profit now depends on z
    z = params["spot_rate"]

    # price(z) = price_per_fleet_unit * (z / spot_rate)
    price_z = params["price_per_fleet_unit"] * (z / params["spot_rate"])

    # Manual computation:
    revenue = z * k
    operating_costs = params["operation_cost"] * k + params["fixed_cost"]
    depreciation_cost = params["depreciation"] * price_z * k
    expected = revenue - operating_costs - depreciation_cost

    p1 = profit(k, z, params)

    assert abs(p1 - expected) < 1e-9


def test_profit_increases_with_k():
    params = load_params()

    z = params["spot_rate"]   # use mean spot rate to avoid noise

    p1 = profit(1, z, params)
    p2 = profit(2, z, params)

    assert p2 > p1   # holding constant z, more trucks => more revenue
