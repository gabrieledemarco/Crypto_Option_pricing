"""
Streamlit app for analyzing options on cryptocurrencies.
It estimates the fair value of Call and Put options via Monte Carlo simulations (with t-Student distribution)
and compares them to Black-Scholes pricing, showing descriptive stats, distribution fitting,
interactive plots, and export options.
"""

import datetime
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, t

from option_analysis.price_data_downloader import PriceDataDownloader
from option_analysis.return_analyzer import ReturnAnalyzer
from option_analysis.option_pricer import OptionPricer

st.set_page_config(page_title="Crypto Option Analysis", layout="wide")
st.title("📈 Crypto Option Pricing")
st.subheader("Using t-Student vs Normal Distribution")

st.markdown("""
This app performs a **quantitative options analysis** on crypto assets (e.g., SOL-USD) including:

- 📊 Statistical analysis of historical returns  
- 📉 Fit with **t-Student distribution** and comparison with the normal distribution  
- 📈 Call/Put option pricing via Monte Carlo simulation  
- 📘 Comparison with Black-Scholes pricing  
- 🎯 Visualizations of price paths and option payoff distribution  
""")

# Sidebar
st.sidebar.header("⚙️ Parameters")
asset = st.sidebar.text_input("Underlying asset", "SOL-USD")
r = st.sidebar.number_input("Risk-Free Rate (%)", value=6.8) / 100
expiry = st.sidebar.date_input("Expiration date", value=datetime.date.today() + datetime.timedelta(days=12))
today = datetime.date.today()
T_days = (expiry - today).days
spot = 100
strike = st.sidebar.number_input("Strike Price", value=float(spot), step=0.01)
run_analysis = st.sidebar.button("▶️ RUN Analysis")
st.sidebar.markdown("---")

if run_analysis:
    if T_days <= 0:
        st.error("⚠️ Expiration date must be in the future.")
        st.stop()

    st.markdown("---")
    with st.spinner("⌛ Loading data..."):
        st.subheader("1️⃣ Historical Data & Return Analysis")
        try:
            downloader = PriceDataDownloader(symbol=asset)
            df = downloader.fetch_data()
        except (ValueError, ConnectionError) as e:
            st.error(f"❌ Error while downloading data: {e}")
            st.stop()

        if df.empty:
            st.error("❌ No data available for the selected symbol.")
            st.stop()

        spot = float(df['close'].iloc[-1])

        analyzer = ReturnAnalyzer(df)
        stats = analyzer.descriptive_stats()
        df_t, loc, scale = analyzer.fit_student_t()
        returns = analyzer.get_filtered_returns()

        col1, col2 = st.columns(2)
        with col1:
            st.write("**Descriptive Statistics:**")
            st.latex(r"\mu = \mathbb{E}[r_t], \quad \sigma = \sqrt{\mathbb{E}[(r_t - \mu)^2]}")
            st.json(stats)

            st.write("**Estimated t-Student distribution (fit on log returns):**")
            st.latex(
                r"f(x; \, \nu, \mu, \sigma) = "
                r"\frac{\Gamma\left(\frac{\nu+1}{2}\right)}"
                r"{\sqrt{\nu\pi} \, \Gamma\left(\frac{\nu}{2}\right)\sigma} "
                r"\left(1 + \frac{(x - \mu)^2}{\nu\sigma^2} \right)^{-\frac{\nu+1}{2}}"
            )

            col_m1, col_m2, col_m3 = st.columns(3)
            with col_m1:
                st.latex(fr"\nu = {df_t:.2f}")
            with col_m2:
                st.latex(fr"\mu = {loc:.6f}")
            with col_m3:
                st.latex(fr"\sigma = {scale:.6f}")

        with col2:
            fig, ax = plt.subplots(figsize=(10, 6))
            sns.histplot(returns, bins=100, stat='density', label="Empirical", color='skyblue', ax=ax)
            x = np.linspace(returns.min(), returns.max(), 500)
            ax.plot(x, norm.pdf(x, returns.mean(), returns.std()), 'r--', label='Normal')
            ax.plot(x, t.pdf(x, df_t, loc, scale), 'g-', label=f't-Student (df={df_t:.1f})')
            ax.set_title("Distribution Fit")
            ax.legend()
            st.pyplot(fig)

            # === QQ Plot personalizzati
            # st.subheader("📐 QQ Plot: confronto con distribuzioni teoriche")

            colqq1, colqq2 = st.columns(2)
            with colqq1:
                fig_qq1, ax_qq1 = plt.subplots(figsize=(3, 3))
                analyzer.qq_plot_manual(dist='t',
                                        dist_params=(df_t, loc, scale),
                                        ax=ax_qq1,
                                        title="QQ Plot vs t-Student")
                st.pyplot(fig_qq1)

            with colqq2:
                fig_qq2, ax_qq2 = plt.subplots(figsize=(3, 3))
                analyzer.qq_plot_manual(dist='norm', ax=ax_qq2, title="QQ Plot vs Normale")
                st.pyplot(fig_qq2)

        st.markdown("---")

        st.subheader("2️⃣ Option Valuation: Call & Put")
        pricer = OptionPricer(spot, strike, r, T_days, scale, df_t)
        call_price, call_prob, final_prices = pricer.monte_carlo_price('call')
        put_price, put_prob, _ = pricer.monte_carlo_price('put')

        bs_call = pricer.black_scholes_price(stats['vol_annualized'], 'call')
        bs_put = pricer.black_scholes_price(stats['vol_annualized'], 'put')

        bs_call_prob_itm = pricer.probability_itm_bs(stats['vol_annualized'], 'call')
        bs_put_prob_itm = pricer.probability_itm_bs(stats['vol_annualized'], 'put')

        col11, col12 = st.columns(2)
        with col11:
            st.metric("Monte Carlo CALL (t-Student)", f"{call_price:.4f} USDT")
            st.write(f"📈 CALL ITM probability (t-Student): `{call_prob:.2%}`")
            st.metric("Black-Scholes CALL (BS)", f"{bs_call:.4f} USDT")
            st.write(f"🧠 Probabilità CALL ITM (BS): `{bs_call_prob_itm:.2%}`")

        with col12:
            st.metric("Monte Carlo PUT (t-Student)", f"{put_price:.4f} USDT")
            st.write(f"📉 PUT ITM probability (t-Student): `{put_prob:.2%}`")
            st.metric("Black-Scholes PUT (BS)", f"{bs_put:.4f} USDT")
            st.write(f"🧠 Probabilità PUT ITM (BS): `{bs_put_prob_itm:.2%}`")

        st.markdown("**💡 How to interpret the results:**")
        st.markdown("- Monte Carlo uses the empirical distribution of returns (t-Student).")
        st.markdown("- Black-Scholes assumes lognormal returns with constant volatility.")
        st.markdown("- Probabilities indicate the chance of the option being **in the money (ITM)** at expiry.")

        st.markdown("---")

        # === Option Greeks Section ===
        st.subheader("📉 Option Greeks")

        col_greek_bs, col_greek_mc = st.columns(2)
        print(type(stats['vol_annualized']))
        with col_greek_bs:
            st.markdown("**Greeks – Black-Scholes**")
            bs_call_greeks = pricer.compute_greeks_bs(option_type='call',
                                                      sigma_annual=(stats['vol_annualized']))
            bs_put_greeks = pricer.compute_greeks_bs(option_type='put',
                                                     sigma_annual=float(stats['vol_annualized']))
            st.dataframe(pd.DataFrame({
                "Greek": list(bs_call_greeks.keys()),
                "Call": list(bs_call_greeks.values()),
                "Put": list(bs_put_greeks.values())
            }))

        with col_greek_mc:
            st.markdown("**Greeks – Monte Carlo (t-Student)**")
            mc_call_greeks = pricer.compute_greeks_mc('call')
            mc_put_greeks = pricer.compute_greeks_mc('put')
            st.dataframe(pd.DataFrame({
                "Greek": list(mc_call_greeks.keys()),
                "Call": list(mc_call_greeks.values()),
                "Put": list(mc_put_greeks.values())
            }))

        # Quick reference
        st.markdown("### 📘 Option Greeks – Quick Reference")
        st.markdown("""
        - **Delta (Δ)**: Measures how much the option's price changes for a $1 move in the underlying asset.  
          Call deltas range from 0 to 1, put deltas from -1 to 0.

        - **Gamma (Γ)**: Describes how fast Delta changes when the underlying price changes.  
          High gamma = more sensitivity, especially at-the-money.

        - **Theta (Θ)**: Quantifies time decay — how much value the option loses each day.  
          It is typically negative for long options.

        - **Vega (ν)**: Indicates how much the option's value changes with a 1% change in implied volatility.

        - **Rho (ρ)**: Measures how the option's price changes with a 1% shift in the risk-free interest rate.
        """)

        st.markdown("---")
        st.subheader("📊 Monte Carlo Simulation Analysis")
        np.random.seed(42)
        returns_matrix = np.random.standard_t(df_t, size=(1000, T_days)) * scale
        log_returns_shifted = np.zeros((returns_matrix.shape[0], returns_matrix.shape[1] + 1))
        log_returns_shifted[:, 1:] = np.cumsum(returns_matrix, axis=1)
        log_paths = log_returns_shifted + np.log(spot)
        price_paths = np.exp(log_paths)

        fig_cloud, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [3, 1]},
                                             constrained_layout=True)
        for path in price_paths[:200]:
            ax1.plot(range(T_days + 1), path, alpha=0.1, color='steelblue')
        ax1.set_title("Simulated Price Paths (t-Student)")
        ax1.set_xlabel("Days")
        ax1.set_ylabel("Price")
        ax1.grid(True)

        sns.histplot(y=final_prices, bins=3000, ax=ax2, color='orange')
        ax2.axhline(strike, color='red', linestyle='--', label='Strike')
        ax2.set_title("Final Price Distribution")
        ax2.set_xlabel("Density")
        ax2.set_ylabel("Final Price")
        ax2.legend()
        ax2.grid(True)

        ax1.set_ylim(spot * 0.3, spot * 2)
        ax2.set_ylim(spot * 0.3, spot * 2)

        st.pyplot(fig_cloud)

        # Dopo i calcoli principali
        gbm_paths = pricer.simulate_gbm_paths(sigma_annual=stats['vol_annualized'], n_sim=1000)
        fig_gbm = pricer.plot_gbm_paths_and_distribution(gbm_paths)
        st.pyplot(fig_gbm)


        st.markdown("---")
        st.subheader("📥 Export Data")
        df_download = pd.DataFrame({"timestamp": df.index, "close": df["close"].squeeze()})
        col111, col112 = st.columns(2)
        with col111:
            st.download_button("📁 Download historical prices", data=df_download.to_csv(index=False),
                               file_name="historical_prices.csv")

            result_df = pd.DataFrame({
                "Method": ["Monte Carlo", "Black-Scholes"],
                "Call Price": [call_price, bs_call],
                "Put Price": [put_price, bs_put],
                "Call Prob Profit": [call_prob, None],
                "Put Prob Profit": [put_prob, None]
            })
        with col112:
            st.download_button("📁 Download option results", data=result_df.to_csv(index=False),
                               file_name="option_results.csv")

        st.success("Analysis completed ✅")

st.markdown("---")
st.markdown("""
<div style='text-align: center;'>

📘 This project is open-source!  
Code available on [GitHub](https://github.com/gabrieledemarco/Crypto_Option_pricing).

<br>

<a href="https://www.buymeacoffee.com/Gabridemarco95?new=1" target="_blank">
    <img src="https://img.buymeacoffee.com/button-api/?text=Buy me a coffee&emoji=☕&slug=Gabridemarco95&button_colour=FFDD00&font_colour=000000&font_family=Arial&outline_colour=000000&coffee_colour=ffffff" />
</a>

<br><br>

📬 Contacts:  
[LinkedIn](https://www.linkedin.com/in/gabriele-de-marco-17a02ba7/) | 
[Email](mailto:gabridemarco091@gmail.com)  
© 2025 Gabriele De Marco. All rights reserved.

<br><br>

<img src="https://img.shields.io/badge/-Python-3776AB?logo=python&logoColor=white">
<img src="https://img.shields.io/badge/-Streamlit-FF4B4B?logo=streamlit&logoColor=white">
<img src="https://img.shields.io/badge/-SciPy-8CAAE6?logo=scipy&logoColor=white">
<img src="https://img.shields.io/badge/-NumPy-013243?logo=numpy&logoColor=white">
<img src="https://img.shields.io/badge/-Pandas-150458?logo=pandas&logoColor=white">

</div>
""", unsafe_allow_html=True)
