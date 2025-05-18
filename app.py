# app.py
import streamlit as st
import datetime
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm, t
from option_analysis.price_data_downloader import PriceDataDownloader
from option_analysis.return_analyzer import ReturnAnalyzer
from option_analysis.option_pricer import OptionPricer

st.set_page_config(page_title="Opzione su Crypto", layout="wide")
st.title("📈 Analisi Opzioni Monte Carlo su Cripto con t-Student")
spot = 1
# === Sidebar Input ===
st.sidebar.header("⚙️ Parametri")
asset = st.sidebar.text_input("Sottostante (asset)", "SOLUSDT")  # future: espandibile
r = st.sidebar.number_input("Tasso Risk-Free (%)", value=6.8) / 100
expiry = st.sidebar.date_input("Data di scadenza", value=datetime.date.today() + datetime.timedelta(days=12))
today = datetime.date.today()
T_days = (expiry - today).days
strike = st.sidebar.number_input("Strike Price", value=round(spot))
run_analysis = st.sidebar.button("▶️ RUN Analisi")

st.sidebar.markdown("---")

if run_analysis:
    if T_days <= 0:
        st.error("⚠️ La data di scadenza deve essere nel futuro.")
        st.stop()

    # === Fase 1: Scarica dati e calcola rendimenti ===
    st.subheader("1️⃣ Dati Storici e Analisi dei Rendimenti")
    downloader = PriceDataDownloader(symbol=asset)
    df = downloader.fetch_binance_data()

    analyzer = ReturnAnalyzer(df)
    stats = analyzer.descriptive_stats()
    df_t, loc, scale = analyzer.fit_student_t()
    returns = analyzer.get_filtered_returns()
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Statistiche Descrittive:**")
        st.json(stats)

        st.write("**Distribuzione t-student stimata:**")
        st.markdown(f"- Gradi di libertà (df): `{df_t:.2f}`\n- Media (loc): `{loc:.6f}`\n- Deviazione (scale): `{scale:.6f}`")

    with col2:
    # Grafico: Fit t-student vs Normale
        fig, ax = plt.subplots(figsize=(20, 12))
        sns.histplot(returns, bins=100, stat='density', label="Empirica", color='skyblue', ax=ax)
        x = np.linspace(returns.min(), returns.max(), 500)
        ax.plot(x, norm.pdf(x, returns.mean(), returns.std()), 'r--', label='Normale')
        ax.plot(x, t.pdf(x, df_t, loc, scale), 'g-', label=f't-Student (df={df_t:.1f})')
        ax.set_title("Fit della Distribuzione Empirica")
        ax.legend()
        st.pyplot(fig)

    # === Fase 2: Calcolo opzioni ===
    st.subheader("2️⃣ Valutazione Opzione Call e Put")

    spot = df['close'].iloc[-1]


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


    # === Grafici percorsi simulati e distribuzione finale ===
    st.subheader("📊 Analisi Grafica Monte Carlo")

    # Ricrea simulazioni per cloud plot
    np.random.seed(42)
    returns_matrix = np.random.standard_t(df_t, size=(1000, T_days)) * scale

    # Costruzione percorsi logaritmici partendo da 0
    log_returns_shifted = np.zeros((returns_matrix.shape[0], returns_matrix.shape[1] + 1))
    log_returns_shifted[:, 1:] = np.cumsum(returns_matrix, axis=1)

    # Trasformazione in prezzi reali, partendo da log(spot)
    log_paths = log_returns_shifted + np.log(spot)
    price_paths = np.exp(log_paths)

    # === Grafico affiancato ===
    fig_cloud, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5), gridspec_kw={'width_ratios': [3, 1]},
                                         constrained_layout=True)

    # A sinistra: nuvola di percorsi simulati
    for path in price_paths[:200]:
        ax1.plot(range(T_days + 1), path, alpha=0.1, color='steelblue')
    ax1.set_title("Percorsi simulati del prezzo")
    ax1.set_xlabel("Giorni")
    ax1.set_ylabel("Prezzo SOL/USDT")
    ax1.grid(True)

    sns.histplot(y=final_prices, bins=100, ax=ax2, color='orange')
    ax2.set_title("Distribuzione prezzi finali")
    ax2.set_xlabel("Densità")
    ax2.set_ylabel("Prezzo finale")
    ax2.grid(True)

    ax2.legend()
    common_ylim = (min(price_paths.min(), final_prices.min()), max(price_paths.max(), final_prices.max()))
    #ax1.set_ylim(common_ylim)
    #ax2.set_ylim(common_ylim)
    ax1.set_ylim(80, 400)
    ax2.set_ylim(80, 400)

    st.pyplot(fig_cloud)

    # === CSV download ===
    st.subheader("📥 Esporta Dati")
    df_download = pd.DataFrame({"timestamp": df.index, "close": df["close"]})
    st.download_button("📁 Scarica prezzi storici", data=df_download.to_csv(index=False), file_name="prezzi_storici.csv")

    result_df = pd.DataFrame({
        "Metodo": ["Monte Carlo", "Black-Scholes"],
        "Call Price": [call_price, bs_call],
        "Put Price": [put_price, bs_put],
        "Call Prob Profit": [call_prob, None],
        "Put Prob Profit": [put_prob, None]
    })
    st.download_button("📁 Scarica risultati opzione", data=result_df.to_csv(index=False), file_name="risultati_opzione.csv")

    st.success("Analisi completata ✅")