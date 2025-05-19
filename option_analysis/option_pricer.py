"""
Modulo per il pricing di opzioni con metodi Monte Carlo e Black-Scholes.
"""

import numpy as np
from scipy.stats import norm


class OptionPricer:
    def __init__(self, spot, strike, r, T_days, sigma_daily, df_t, n_sim=100_000):
        self.S0 = spot
        self.K = strike
        self.r = r
        self.days_to_expiry = days_to_expiry
        self.expiry_in_years = days_to_expiry / 365
        self.sigma = sigma_daily
        self.df_t = df_t
        self.n_sim = n_sim

    def simulate_prices(self):
        """Simula i prezzi futuri usando la distribuzione t-Student."""
        np.random.seed(42)
        returns = np.random.standard_t(self.df_t, size=(self.n_sim, self.days_to_expiry)) * self.sigma
        log_paths = np.cumsum(returns, axis=1)
        log_paths = np.insert(log_paths, 0, 0, axis=1)
        return np.exp(log_paths + np.log(self.spot_price))[:, -1]

    def monte_carlo_price(self, option_type='call'):
        """Calcola il prezzo di una call o put usando Monte Carlo."""
        final_prices = self.simulate_prices()
        if option_type == 'call':
            payoff = np.maximum(final_prices - self.strike_price, 0)
        else:
            payoff = np.maximum(self.strike_price - final_prices, 0)
        discounted = np.exp(-self.r * self.expiry_in_years) * np.mean(payoff)
        prob_profit = np.mean(payoff > 0)
        return discounted, prob_profit, final_prices

    def black_scholes_price(self, sigma_annual, option_type='call'):
        """Calcola il prezzo dell'opzione usando la formula di Black-Scholes."""
        d1 = (np.log(self.spot_price / self.strike_price) +
              (self.r + 0.5 * sigma_annual ** 2) * self.expiry_in_years) / (sigma_annual * np.sqrt(self.expiry_in_years))
        d2 = d1 - sigma_annual * np.sqrt(self.expiry_in_years)
        if option_type == 'call':
            return self.spot_price * norm.cdf(d1) - self.strike_price * np.exp(-self.r * self.expiry_in_years) * norm.cdf(d2)
        return self.strike_price * np.exp(-self.r * self.expiry_in_years) * norm.cdf(-d2) - self.spot_price * norm.cdf(-d1)
