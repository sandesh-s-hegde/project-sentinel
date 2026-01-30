import numpy as np
from scipy.stats import norm

def calculate_black_scholes(S, K, T, r, sigma):
    """
    Black-Scholes Call Option Valuation.

    Args:
        S: Asset Value (Current value of capacity/inventory)
        K: Exercise Price (Cost to deploy/expand)
        T: Time to Maturity (Years)
        r: Risk-free interest rate
        sigma: Volatility (The input from Computer Vision)
    """
    # If volatility is effectively zero, the value is just Intrinsic Value
    if sigma <= 0.01:
        return max(S - K, 0)

    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)

    call_price = S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2)
    return call_price

def run_simulation(volatility_data, S, K, T, r):
    """
    Compare Dynamic (Vision) vs Static (Accounting) pricing.
    Returns:
        dynamic_prices: Time-series using vision-based sigma.
        static_prices: Time-series using mean sigma.
        static_sigma: The mean sigma value used.
    """
    # Static baseline = Mean of observed volatility from the video
    # This simulates a manager who only checks risk "once a year" (average)
    static_sigma = np.mean(volatility_data) if len(volatility_data) > 0 else 0.2

    dynamic_prices = [calculate_black_scholes(S, K, T, r, sig) for sig in volatility_data]
    static_prices = [calculate_black_scholes(S, K, T, r, static_sigma) for _ in volatility_data]

    return dynamic_prices, static_prices, static_sigma