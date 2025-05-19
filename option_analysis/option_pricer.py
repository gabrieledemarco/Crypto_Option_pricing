"""Modulo per il pricing di opzioni con metodi Monte Carlo e Black-Scholes, incluse le Greeks.
"""

import numpy as np
from scipy.stats import norm


class OptionPricer:
    """Classe per la valutazione di opzioni call e put."""

    def __init__(self, spot_price, strike_price, r, days_to_expiry, sigma_daily, df_t, n_sim=100_000):
        """Inizializza i parametri per la simulazione."""
        self.spot_price = spot_price
        self.strike_price = strike_price
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

    def simulate_gbm_paths(self, sigma_annual, n_sim=1000):
        """Simula traiettorie di prezzo secondo moto geometrico browniano (GBM)."""
        np.random.seed(42)
        T = self.expiry_in_years
        dt = T / self.days_to_expiry
        paths = np.zeros((n_sim, self.days_to_expiry + 1))
        paths[:, 0] = self.spot_price
        for t in range(1, self.days_to_expiry + 1):
            z = np.random.normal(0, 1, n_sim)
            paths[:, t] = paths[:, t - 1] * np.exp(
                (self.r - 0.5 * sigma_annual ** 2) * dt + sigma_annual * np.sqrt(dt) * z
            )
        return paths

    def plot_gbm_paths_and_distribution(self, gbm_paths):
        """Genera il grafico dei percorsi GBM e distribuzione finale."""
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [3, 1]},
                                       constrained_layout=True)

        for path in gbm_paths[:200]:
            ax1.plot(range(gbm_paths.shape[1]), path, alpha=0.1, color='steelblue')
        ax1.set_title("Simulated Price Paths (GBM - STD Normal distribution)")
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Price")
        ax1.grid(True)

        final_prices = gbm_paths[:, -1]
        sns.histplot(y=final_prices, bins=300, ax=ax2, color='orange')
        ax2.axhline(self.strike_price, color='red', linestyle='--', label='Strike')
        ax2.set_title("Final Price Distribution")
        ax2.set_xlabel("Density")
        ax2.set_ylabel("Final Price")
        ax2.legend()
        ax2.grid(True)

        ax1.set_ylim(self.spot_price * 0.3, self.spot_price * 2)
        ax2.set_ylim(self.spot_price * 0.3, self.spot_price * 2)

        return fig

    def black_scholes_price(self, sigma_annual, option_type='call'):
        """Calcola il prezzo dell'opzione usando la formula di Black-Scholes."""
        d1 = (np.log(self.spot_price / self.strike_price) +
              (self.r + 0.5 * sigma_annual ** 2) * self.expiry_in_years) / (sigma_annual * np.sqrt(self.expiry_in_years))
        d2 = d1 - sigma_annual * np.sqrt(self.expiry_in_years)
        if option_type == 'call':
            return self.spot_price * norm.cdf(d1) - self.strike_price * np.exp(-self.r * self.expiry_in_years) * norm.cdf(d2)
        return self.strike_price * np.exp(-self.r * self.expiry_in_years) * norm.cdf(-d2) - self.spot_price * norm.cdf(-d1)

    def compute_greeks_bs(self, sigma_annual, option_type='call'):
        """Calcola le Greeks (Delta, Gamma, Vega, Theta, Rho) con Black-Scholes."""

        S, K, r, T, sigma = self.spot_price, self.strike_price, self.r, self.expiry_in_years, sigma_annual
        d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
        d2 = d1 - sigma * np.sqrt(T)

        greeks = {}
        if option_type == 'call':
            greeks['Delta'] = norm.cdf(d1)
            greeks['Theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) -
                               r * K * np.exp(-r * T) * norm.cdf(d2)) / 365
            greeks['Rho'] = K * T * np.exp(-r * T) * norm.cdf(d2) / 100
        else:
            greeks['Delta'] = -norm.cdf(-d1)
            greeks['Theta'] = (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T)) +
                               r * K * np.exp(-r * T) * norm.cdf(-d2)) / 365
            greeks['Rho'] = -K * T * np.exp(-r * T) * norm.cdf(-d2) / 100

        greeks['Gamma'] = norm.pdf(d1) / (S * sigma * np.sqrt(T))
        greeks['Vega'] = S * norm.pdf(d1) * np.sqrt(T) / 100

        return greeks

    def compute_greeks_mc(self, option_type='call', bump_size=0.01):
        """Stima le Greeks (Delta, Gamma, Vega, Rho, Theta) via Monte Carlo."""
        base_price, _, _ = self.monte_carlo_price(option_type)

        self.spot_price += bump_size
        up_price, _, _ = self.monte_carlo_price(option_type)

        self.spot_price -= 2 * bump_size
        down_price, _, _ = self.monte_carlo_price(option_type)
        self.spot_price += bump_size

        delta = (up_price - down_price) / (2 * bump_size)
        gamma = (up_price - 2 * base_price + down_price) / (bump_size ** 2)

        self.sigma += 0.01
        vega_up, _, _ = self.monte_carlo_price(option_type)
        vega = (vega_up - base_price) / 0.01
        self.sigma -= 0.01

        self.r += 0.01
        rho_up, _, _ = self.monte_carlo_price(option_type)
        rho = (rho_up - base_price) / 0.01
        self.r -= 0.01

        self.days_to_expiry -= 1
        self.expiry_in_years = self.days_to_expiry / 365
        theta_down, _, _ = self.monte_carlo_price(option_type)
        theta = base_price - theta_down
        self.days_to_expiry += 1
        self.expiry_in_years = self.days_to_expiry / 365

        return {
            'Delta': delta,
            'Gamma': gamma,
            'Vega': vega,
            'Rho': rho,
            'Theta': theta
        }

    def probability_itm_bs(self, sigma_annual, option_type):
        S = float(self.spot_price)
        K = float(self.strike_price)
        T = float(self.expiry_in_years)
        r = float(self.r)
        sigma = float(sigma_annual)

        d2 = (np.log(S / K) + (r - 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))

        if option_type == 'put':
            return 1 - norm.cdf(d2)
        elif option_type == 'call':
            return norm.cdf(d2)
        else:
            raise ValueError("option_type must be 'call' or 'put'")