import numpy as np
from scipy.stats import norm

class OptionPricer:
    def __init__(self, spot, strike, r, T_days, sigma_daily, df_t, n_sim=100_000):
        self.S0 = spot
        self.K = strike
        self.r = r
        self.T_days = T_days
        self.T = T_days / 365
        self.sigma = sigma_daily
        self.df_t = df_t
        self.n_sim = n_sim

    def simulate_prices(self):
        np.random.seed(42)
        returns = np.random.standard_t(self.df_t, size=(self.n_sim, self.T_days)) * self.sigma
        log_paths = np.cumsum(returns, axis=1)
        log_paths = np.insert(log_paths, 0, 0, axis=1)
        return np.exp(log_paths + np.log(self.S0))[:, -1]

    def monte_carlo_price(self, option_type='call'):
        final_prices = self.simulate_prices()
        if option_type == 'call':
            payoff = np.maximum(final_prices - self.K, 0)
        else:
            payoff = np.maximum(self.K - final_prices, 0)
        discounted = np.exp(-self.r * self.T) * np.mean(payoff)
        prob_profit = np.mean(payoff > 0)
        return discounted, prob_profit, final_prices

    def black_scholes_price(self, sigma_annual, option_type='call'):
        d1 = (np.log(self.S0 / self.K) + (self.r + 0.5 * sigma_annual ** 2) * self.T) / (sigma_annual * np.sqrt(self.T))
        d2 = d1 - sigma_annual * np.sqrt(self.T)
        if option_type == 'call':
            return self.S0 * norm.cdf(d1) - self.K * np.exp(-self.r * self.T) * norm.cdf(d2)
        else:
            return self.K * np.exp(-self.r * self.T) * norm.cdf(-d2) - self.S0 * norm.cdf(-d1)
