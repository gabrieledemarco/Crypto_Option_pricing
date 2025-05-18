import numpy as np
from scipy.stats import t, jarque_bera, shapiro

class ReturnAnalyzer:
    def __init__(self, df_close):
        self.df = df_close.copy()
        self.df["log_return"] = np.log(self.df["close"] / self.df["close"].shift(1))
        self.filtered = self.df["log_return"].dropna()
        self.filtered = self.filtered[(self.filtered > -1) & (self.filtered < 1)]

    def descriptive_stats(self):
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
        jb = jarque_bera(self.filtered)
        sw = shapiro(self.filtered)

        try:
            msg_jb = {
                "stat": jb.statistic, "p": jb.pvalue,
                "skew": jb.skewness, "kurt": jb.kurtosis
            }
        except:
            msg_jb = {
                "stat": jb[0], "p": jb[1], "skew": jb[2], "kurt": jb[3]
            }

        try:
            msg_sw = {"stat": sw.statistic, "p": sw.pvalue}
        except:
            msg_sw = {"stat": sw[0], "p": sw[1]}

        return {
            "jarque_bera": msg_jb,
            "shapiro": msg_sw
        }

    def fit_student_t(self):
        df_t, loc, scale = t.fit(self.filtered)
        return df_t, loc, scale

    def get_filtered_returns(self):
        return self.filtered
