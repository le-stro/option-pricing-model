import numpy as np
from scipy.stats import norm


def _d1(S, K, T, r, sigma):
    return (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))


def _d2(S, K, T, r, sigma):
    return _d1(S, K, T, r, sigma) - sigma * np.sqrt(T)


def call_price(S, K, T, r, sigma):
    """
    European call option price via Black-Scholes.

    Params:
        S     : current stock price
        K     : strike price
        T     : time to expiry in years (e.g. 3 months = 0.25)
        r     : risk-free interest rate (annualized, e.g. 0.05 for 5%)
        sigma : annualized volatility (e.g. 0.2 for 20%)

    Returns: call price (float)
    """
    if T <= 0:
        return max(S - K, 0)  # at expiry: intrinsic value only

    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    return S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)


def put_price(S, K, T, r, sigma):
    """
    European put option price via put-call parity:
        P = C - S + K * e^(-rT)

    Same params as call_price.
    Returns: put price (float)
    """
    if T <= 0:
        return max(K - S, 0)

    return call_price(S, K, T, r, sigma) - S + K * np.exp(-r * T)
