from option_analysis.price_data_downloader import PriceDataDownloader
from option_analysis.return_analyzer import ReturnAnalyzer
from option_analysis.option_pricer import OptionPricer
from option_analysis.strike_sensitivity import StrikeSensitivity

# === Step 1: Download data ===
downloader = PriceDataDownloader()
df = downloader.fetch_binance_data()

# === Step 2: Analyze returns ===
analyzer = ReturnAnalyzer(df)
stats = analyzer.descriptive_stats()
print("\n--- Statistiche Descrittive ---")
for k, v in stats.items():
    print(f"{k}: {v:.6f}")

normality = analyzer.normality_tests()
print("\n--- Test di Normalità ---")
print(normality)

df_t, loc, scale = analyzer.fit_student_t()
print("\nDistribuzione t-student: df=", df_t, " loc=", loc, " scale=", scale)

# === Step 3: Calcolo opzione ===
spot = df['close'].iloc[-1]
strike = 160
r = 0.068
T_days = 12

pricer = OptionPricer(spot, strike, r, T_days, scale, df_t)
mc_price, prob_profit, _ = pricer.monte_carlo_price('call')
bs_price = pricer.black_scholes_price(stats['vol_annualized'], 'call')

print("\nMonte Carlo fair value:", mc_price)
print("Probabilità di profitto:", prob_profit)
print("Black-Scholes fair value:", bs_price)

# === Step 4: Sensibilità per strike ===
sensitivity = StrikeSensitivity(spot, r, T_days, scale, df_t)
results = sensitivity.analyze_strikes([140, 150, 160, 170, 180])
print("\n--- Analisi Sensibilità Strike ---")
print(results)
