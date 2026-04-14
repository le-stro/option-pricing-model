import numpy as np
from black_scholes import call_price, put_price
from greeks import vega


# fallback if vega is near zero (deep in/out of the money)
_VEGA_THRESHOLD = 1e-10
_MAX_ITERATIONS = 100
_TOLERANCE = 1e-6


def implied_volatility(market_price, S, K, T, r, option="call"):
    """
    Recover the volatility implied by a market option price using Newton-Raphson.

    The market price tells us what sigma the market is "pricing in" —
    this is what traders actually quote and compare, not the option price itself.

    Params:
        market_price : observed option price in the market

    Returns: implied volatility (float), or None if it fails to converge.
    """
    pricing_fn = call_price if option == "call" else put_price

    # initial guess: flat 20% vol, reasonable starting point for most equities
    sigma = 0.2

    for _ in range(_MAX_ITERATIONS):
        price = pricing_fn(S, K, T, r, sigma)
        v = vega(S, K, T, r, sigma) * 100  # undo the /100 convention from greeks.py

        if abs(v) < _VEGA_THRESHOLD:
            return None  # option too far in/out of the money to converge

        # Newton-Raphson step: sigma_new = sigma - f(sigma) / f'(sigma)
        sigma -= (price - market_price) / v

        if abs(price - market_price) < _TOLERANCE:
            return sigma

    return None  # did not converge


def volatility_smile(market_prices, strikes, S, T, r, option="call"):
    """
    Compute implied vol for a range of strikes at fixed S and T.
    Reveals the volatility smile / skew — a known Black-Scholes failure mode.

    Params:
        market_prices : list of observed prices, one per strike
        strikes       : list of strike prices (same length)

    Returns: list of implied vols (None where convergence failed)
    """
    return [
        implied_volatility(price, S, k, T, r, option)
        for price, k in zip(market_prices, strikes)
    ]
