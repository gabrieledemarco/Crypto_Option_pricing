"""
App Streamlit per l'analisi di opzioni su criptovalute.
Permette di stimare prezzi di opzioni Call e Put tramite simulazioni Monte Carlo (t-Student)
e confronto con il modello Black-Scholes, mostrando statistiche descrittive, fit distribuzionale,
grafici interattivi e possibilità di esportazione.
"""

# Standard library
import datetime

# Third-party
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, t

# Local imports
from option_analysis.price_data_downloader import PriceDataDownloader
from option_analysis.return_analyzer import ReturnAnalyzer
from option_analysis.option_pricer import OptionPricer

st.set_page_config(page_title="Opzione su Crypto", layout="wide")
st.title("📈 Analisi Opzioni Monte Carlo su Cripto con t-Student")

st.markdown("""
Questa applicazione esegue un'**analisi quantitativa delle opzioni** su asset crypto (es. SOL-USD) tramite:

- 📊 Calcolo statistico dei rendimenti storici
- 📉 Fit con distribuzione **t-Student** e confronto con la normale
- 📈 Pricing opzioni **Call/Put** via simulazione Monte Carlo
- 📘 Calcolo Black-Scholes a confronto
- 🎯 Visualizzazione grafica dei percorsi simulati e distribuzione dei payoff
""")

# === Sidebar Input ===
st.sidebar.header("⚙️ Parametri")
asset = st.sidebar.text_input("Sottostante (asset)", "SOL-USD")
r = st.sidebar.number_input("Tasso Risk-Free (%)", value=6.8) / 100

expiry = st.sidebar.date_input(
    "Data di scadenza",
    value=datetime.date.today() + datetime.timedelta(days=12)
)

today = datetime.date.today()
T_days = (expiry - today).days
spot = 100# noqa: N806
strike = st.sidebar.number_input("Strike Price", value=round(spot))
run_analysis = st.sidebar.button("▶️ RUN Analisi")
st.sidebar.markdown("---")

if run_analysis:
    if T_days <= 0:
        st.error("⚠️ La data di scadenza deve essere nel futuro.")
        st.stop()

    # === Fase 1: Scarica dati e calcola rendimenti ===
    st.subheader("1️⃣ Dati Storici e Analisi dei Rendimenti")
    try:
        downloader = PriceDataDownloader(symbol=asset)
        df = downloader.fetch_data()
    except (ValueError, ConnectionError) as e:
        st.error(f"❌ Errore durante il download dei dati: {e}")
        st.stop()

    if df.empty:
        st.error("❌ Nessun dato disponibile per il simbolo selezionato.")
        st.stop()

    spot = float(df['close'].iloc[-1])
   # strike = st.sidebar.number_input("Strike Price", value=round(spot))

    analyzer = ReturnAnalyzer(df)
    stats = analyzer.descriptive_stats()
    df_t, loc, scale = analyzer.fit_student_t()
    returns = analyzer.get_filtered_returns()

    col1, col2 = st.columns(2)
    with col1:
        st.write("**Statistiche Descrittive:**")
        st.latex(r"\mu = \mathbb{E}[r_t], \quad \sigma = \sqrt{\mathbb{E}[(r_t - \mu)^2]}")
        st.json(stats)

        st.write("**Distribuzione t-student stimata (fit sui log-return):**")
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
        sns.histplot(returns, bins=100, stat='density', label="Empirica", color='skyblue', ax=ax)
        x = np.linspace(returns.min(), returns.max(), 500)
        ax.plot(x, norm.pdf(x, returns.mean(), returns.std()), 'r--', label='Normale')
        ax.plot(x, t.pdf(x, df_t, loc, scale), 'g-', label=f't-Student (df={df_t:.1f})')
        ax.set_title("Fit della Distribuzione Empirica")
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



    # === Fase 2: Calcolo opzioni ===
    st.subheader("2️⃣ Valutazione Opzione Call e Put")
    pricer = OptionPricer(spot, strike, r, T_days, scale, df_t)
    call_price, call_prob, final_prices = pricer.monte_carlo_price('call')
    put_price, put_prob, _ = pricer.monte_carlo_price('put')

    bs_call = pricer.black_scholes_price(stats['vol_annualized'], 'call')
    bs_put = pricer.black_scholes_price(stats['vol_annualized'], 'put')

    col11, col12 = st.columns(2)
    with col11:
        st.metric("Monte Carlo CALL (t-student)", f"{call_price:.4f} USDT")
        st.write(f"📈 Probabilità CALL > 0: `{call_prob:.2%}`")
        st.metric("Black-Scholes CALL", f"{bs_call:.4f} USDT")

    with col12:
        st.metric("Monte Carlo PUT (t-student)", f"{put_price:.4f} USDT")
        st.write(f"📉 Probabilità PUT > 0: `{put_prob:.2%}`")
        st.metric("Black-Scholes PUT", f"{bs_put:.4f} USDT")

    # Interpretazione risultati
    st.markdown("**💡 Interpretazione dei risultati:**")
    st.markdown("- Il prezzo Monte Carlo tiene conto della distribuzione empirica dei rendimenti (t-Student).")
    st.markdown("- Il Black-Scholes assume una distribuzione normale logaritmica e volatilità costante.")
    st.markdown("- Le probabilità rappresentano la stima che l'opzione sia **in guadagno (ITM)** alla scadenza.")

    # === Grafici percorsi simulati e distribuzione finale ===
    st.subheader("📊 Analisi Grafica Monte Carlo")
    np.random.seed(42)
    returns_matrix = np.random.standard_t(df_t, size=(1000, T_days)) * scale

    log_returns_shifted = np.zeros((returns_matrix.shape[0], returns_matrix.shape[1] + 1))
    log_returns_shifted[:, 1:] = np.cumsum(
        returns_matrix,
        axis=1
    )

    log_paths = log_returns_shifted + np.log(spot)
    price_paths = np.exp(log_paths)

    fig_cloud, (ax1, ax2) = plt.subplots(1,
                                         2,
                                         figsize=(14, 5),
                                         gridspec_kw={'width_ratios': [3, 1]},
                                         constrained_layout=True)

    for path in price_paths[:200]:
        ax1.plot(range(T_days + 1), path, alpha=0.1, color='steelblue')
    ax1.set_title("Percorsi simulati del prezzo")
    ax1.set_xlabel("Giorni")
    ax1.set_ylabel("Prezzo")
    ax1.grid(True)

    sns.histplot(y=final_prices, bins=3000, ax=ax2, color='orange')
    ax2.axhline(strike, color='red', linestyle='--', label='Strike')
    ax2.set_title("Distribuzione prezzi finali")
    ax2.set_xlabel("Densità")
    ax2.set_ylabel("Prezzo finale")
    ax2.legend()
    ax2.grid(True)

    ax1.set_ylim(spot * 0.3, spot * 2)
    ax2.set_ylim(spot * 0.3, spot * 2)

    st.pyplot(fig_cloud)

    # === CSV download ===
    st.subheader("📥 Esporta Dati")

    df_download = pd.DataFrame(
        {"timestamp": df.index,
         "close": df["close"].squeeze()
         }
    )
    col111, col112 = st.columns(2)
    with col111:
        st.download_button("📁 Scarica prezzi storici", data=df_download.to_csv(index=False), file_name="prezzi_storici.csv")

        result_df = pd.DataFrame({
            "Metodo": ["Monte Carlo",
                       "Black-Scholes"],
            "Call Price": [call_price,
                           bs_call],
            "Put Price": [put_price,
                          bs_put],
            "Call Prob Profit": [call_prob,
                                 None],
            "Put Prob Profit": [put_prob,
                                None]
        })
    with col112:
        st.download_button("📁 Scarica risultati opzione", data=result_df.to_csv(index=False), file_name="risultati_opzione.csv")

    st.success("Analisi completata ✅")
