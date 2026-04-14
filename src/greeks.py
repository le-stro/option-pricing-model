import numpy as np
from scipy.stats import norm
from black_scholes import _d1, _d2


def delta(S, K, T, r, sigma, option="call"):
    """
    Rate of change of option price w.r.t. underlying price S.
    Call delta in [0, 1], put delta in [-1, 0].
    """
    d1 = _d1(S, K, T, r, sigma)
    if option == "call":
        return norm.cdf(d1)
    return norm.cdf(d1) - 1


def gamma(S, K, T, r, sigma):
    """
    Rate of change of delta w.r.t. S. Identical for calls and puts.
    High gamma = delta changes rapidly = more hedging needed.
    """
    d1 = _d1(S, K, T, r, sigma)
    return norm.pdf(d1) / (S * sigma * np.sqrt(T))


def theta(S, K, T, r, sigma, option="call"):
    """
    Time decay: change in option price per one calendar day.
    Almost always negative — option loses value as expiry approaches.
    Divided by 365 to express as daily loss.
    """
    d1 = _d1(S, K, T, r, sigma)
    d2 = _d2(S, K, T, r, sigma)

    decay = -(S * norm.pdf(d1) * sigma) / (2 * np.sqrt(T))

    if option == "call":
        return (decay - r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
    return (decay + r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365


def vega(S, K, T, r, sigma):
    """
    Sensitivity of option price w.r.t. volatility sigma.
    Identical for calls and puts.
    Divided by 100 to express as price change per 1% move in sigma.
    """
    d1 = _d1(S, K, T, r, sigma)
    return S * norm.pdf(d1) * np.sqrt(T) / 100


def rho(S, K, T, r, sigma, option="call"):
    """
    Sensitivity w.r.t. risk-free rate r.
    Divided by 100 to express as price change per 1% move in r.
    Least impactful greek in practice.
    """
    d2 = _d2(S, K, T, r, sigma)
    if option == "call":
        return K * T * np.exp(-r * T) * norm.cdf(d2) / 100
    return -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100
