import numpy as np


def simulate_paths(S, T, r, sigma, n_simulations=10_000, n_steps=252):
    """
    Simulate stock price paths via Geometric Brownian Motion.

    GBM: S(t) = S(0) * exp((r - 0.5*sigma^2)*t + sigma*W(t))
    where W(t) is a Wiener process (standard Brownian motion).

    Params:
        n_simulations : number of independent paths
        n_steps       : time steps per path (252 = trading days in a year)

    Returns: array of shape (n_steps + 1, n_simulations)
    """
    dt = T / n_steps

    # random shocks: standard normal, shape (n_steps, n_simulations)
    Z = np.random.standard_normal((n_steps, n_simulations))

    # log-returns per step
    log_returns = (r - 0.5 * sigma**2) * dt + sigma * np.sqrt(dt) * Z

    # cumulative product to build paths, prepend S as starting point
    paths = np.zeros((n_steps + 1, n_simulations))
    paths[0] = S
    paths[1:] = S * np.exp(np.cumsum(log_returns, axis=0))

    return paths


def mc_call_price(S, K, T, r, sigma, n_simulations=10_000, n_steps=252):
    """
    European call price via Monte Carlo.
    Payoff at expiry: max(S_T - K, 0), discounted back to today.
    Converges to Black-Scholes price as n_simulations → ∞.

    Returns: (price, std_error)
        std_error indicates simulation uncertainty.
    """
    paths = simulate_paths(S, T, r, sigma, n_simulations, n_steps)
    S_T = paths[-1]  # terminal stock prices

    payoffs = np.maximum(S_T - K, 0)
    discounted = np.exp(-r * T) * payoffs

    price = discounted.mean()
    std_error = discounted.std() / np.sqrt(n_simulations)

    return price, std_error


def mc_put_price(S, K, T, r, sigma, n_simulations=10_000, n_steps=252):
    """
    European put price via Monte Carlo.
    Payoff at expiry: max(K - S_T, 0), discounted back to today.

    Returns: (price, std_error)
    """
    paths = simulate_paths(S, T, r, sigma, n_simulations, n_steps)
    S_T = paths[-1]

    payoffs = np.maximum(K - S_T, 0)
    discounted = np.exp(-r * T) * payoffs

    price = discounted.mean()
    std_error = discounted.std() / np.sqrt(n_simulations)

    return price, std_error
