"""
Payoff functions for the trucking dynamic programming model.

Updated to make vehicle price (used for depreciation) depend on the current spot rate z.

New signature:
    profit(k, z, params)

Notes:
- z is the current spot-rate (in the same units as params["spot_rate"], i.e. thousands).
- price(z) = price_per_fleet_unit * (z / params["spot_rate"])
- Depreciation cost each period = k * depreciation * price(z)
- This file is minimal: it only implements the new depreciation logic and returns
  the immediate operating profit. It does not change any solver or simulation logic.
"""

def profit(k: int, z: float, params: dict) -> float:
    """
    Compute annual profit (in thousands of dollars) for a firm with k trucks
    when the current spot rate is z (thousands).

    Parameters:
        k      : number of trucks (integer)
        z      : current spot-rate (thousands per truck per year)
        params : dictionary loaded from load_params()

    Returns:
        profit_k (float): profit in thousands of dollars
    """

    # parameters
    spot_mean = float(params["spot_rate"])                 # mu (thousands)
    op_cost   = float(params["operation_cost"])            # operating cost per truck (thousands)
    fixed     = float(params["fixed_cost"])                # fixed cost per period (thousands)
    price_base= float(params["price_per_fleet_unit"])      # base price at mu (thousands)
    depreciation = float(params.get("depreciation", 0.0))  # fraction per period

    # compute state-dependent price (symmetric for buy/sell)
    # protect against division by zero (spot_mean should be > 0)
    price_z = price_base * (float(z) / spot_mean)

    revenue_k = float(z) * int(k)
    operating_costs = op_cost * int(k) + fixed

    # Depreciation treated as per-period cost: k * depreciation * price(z)
    depreciation_cost = depreciation * price_z * int(k)

    return revenue_k - operating_costs - depreciation_cost
