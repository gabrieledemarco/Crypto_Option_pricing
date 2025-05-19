"""
Modulo per l'analisi dei rendimenti logaritmici e fitting statistico.
"""

import numpy as np
from scipy.stats import t, jarque_bera, shapiro, norm
import matplotlib.pyplot as plt


class ReturnAnalyzer:
    """Classe per analizzare i rendimenti storici di un asset."""

    def __init__(self, df_close):
        """Inizializza la classe con un DataFrame di prezzi di chiusura."""
        self.df = df_close.copy()
        self.df["log_return"] = np.log(self.df["close"] / self.df["close"].shift(1))
        self.filtered = self.df["log_return"].dropna()
        self.filtered = self.filtered[(self.filtered > -1) & (self.filtered < 1)]

    def descriptive_stats(self):
        """Calcola statistiche descrittive base sui log-return."""
        return {
            "mean_daily": self.filtered.mean(),
            "std_daily": self.filtered.std(),
            "mean_annualized": self.filtered.mean() * 365,
            "vol_annualized": self.filtered.std() * np.sqrt(365),
            "skew": self.filtered.skew(),
            "kurt": self.filtered.kurtosis(),
            "min": self.filtered.min(),
            "max": self.filtered.max(),
        }

    def normality_tests(self):
        """Applica i test di normalità Jarque-Bera e Shapiro-Wilk."""
        jb = jarque_bera(self.filtered)
        sw = shapiro(self.filtered)

        try:
            msg_jb = {
                "stat": jb.statistic, "p": jb.pvalue,
                "skew": jb.skewness, "kurt": jb.kurtosis
            }
        except (AttributeError, TypeError):
            msg_jb = {
                "stat": jb[0], "p": jb[1], "skew": jb[2], "kurt": jb[3]
            }

        try:
            msg_sw = {"stat": sw.statistic, "p": sw.pvalue}
        except (AttributeError, TypeError):
            msg_sw = {"stat": sw[0], "p": sw[1]}

        return {
            "jarque_bera": msg_jb,
            "shapiro": msg_sw
        }

    def fit_student_t(self):
        """Stima i parametri di una t-student: df, loc e scale."""
        df_t, loc, scale = t.fit(self.filtered)
        return df_t, loc, scale

    def get_filtered_returns(self):
        """Restituisce i log-return filtrati (puliti)."""
        return self.filtered

    def qq_plot_manual(self, dist='t', dist_params=None, ax=None, title="QQ Plot"):
        """Genera un QQ plot manuale rispetto a una distribuzione teorica."""
        if ax is None:
            _, ax = plt.subplots()

        data = np.sort(self.filtered)
        n = len(data)
        probs = (np.arange(1, n + 1) - 0.5) / n

        if dist == "norm":
            data = (data - np.mean(data)) / np.std(data)
            theoretical = norm.ppf(probs)
        elif dist == "t":
            if dist_params is None:
                dist_params = self.fit_student_t()
            df_val, loc_val, scale_val = dist_params
            theoretical = t.ppf(probs, df_val, loc=loc_val, scale=scale_val)
        else:
            raise ValueError("Distribuzione non supportata")

        ax.plot(theoretical, data, "o", alpha=0.5)
        ax.plot(theoretical, theoretical, "r--", label="45°")
        ax.set_title(title)
        ax.set_xlabel("Quantili teorici")
        ax.set_ylabel("Quantili empirici")
        ax.grid(True)
        return ax
