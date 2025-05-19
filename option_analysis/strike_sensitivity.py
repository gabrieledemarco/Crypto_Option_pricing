"""
Modulo per analisi di sensitività del prezzo dell'opzione rispetto allo strike.
"""

import numpy as np
import pandas as pd
from option_pricer import OptionPricer


class StrikeSensitivity:
    """Classe per analizzare la sensitività del prezzo delle opzioni rispetto allo strike."""

    def __init__(self, spot_price, rate, days_to_expiry, sigma_daily, df_t):
        """Inizializza i parametri base della simulazione."""
        self.spot_price = spot_price
        self.rate = rate
        self.days_to_expiry = days_to_expiry
        self.sigma = sigma_daily
        self.df_t = df_t

    def analyze_strikes(self, strike_list):
        """
        Calcola il prezzo dell'opzione call per una lista di strike usando:
        - Monte Carlo (t-student)
        - Black-Scholes (normale)
        """
        results = []
        for strike_price in strike_list:
            pricer = OptionPricer(
                spot_price=self.spot_price,
                strike_price=strike_price,
                r=self.rate,
                days_to_expiry=self.days_to_expiry,
                sigma_daily=self.sigma,
                df_t=self.df_t
            )
            mc_price, _, _ = pricer.monte_carlo_price(option_type='call')
            bs_price = pricer.black_scholes_price(
                sigma_annual=self.sigma * np.sqrt(365),
                option_type='call'
            )
            results.append({
                "Strike": strike_price,
                "MC Price": mc_price,
                "BS Price": bs_price
            })
        return pd.DataFrame(results)
