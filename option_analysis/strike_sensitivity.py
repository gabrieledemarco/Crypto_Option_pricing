import pandas as pd
from .option_pricer import OptionPricer

class StrikeSensitivity:
    def __init__(self, spot, r, T_days, sigma_daily, df_t):
        self.spot = spot
        self.r = r
        self.T_days = T_days
        self.sigma = sigma_daily
        self.df_t = df_t

    def analyze_strikes(self, strikes):
        results = []
        for K in strikes:
            pricer = OptionPricer(self.spot, K, self.r, self.T_days, self.sigma, self.df_t)
            mc_price, _, _ = pricer.monte_carlo_price(option_type='call')
            bs_price = pricer.black_scholes_price(self.sigma * np.sqrt(365), option_type='call')
            results.append({"Strike": K, "MC Price": mc_price, "BS Price": bs_price})
        return pd.DataFrame(results)
