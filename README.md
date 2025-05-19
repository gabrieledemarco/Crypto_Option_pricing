
# 📈 Streamlit Option App – Crypto Option Pricing with Monte Carlo & t-Student

An interactive web app to analyze the expected payoff and fair value of European **call/put options** on crypto assets (e.g., SOL/USDT).  
This tool compares **Monte Carlo simulations** using **Student's t-distribution** against the classical **Black-Scholes model**, incorporating statistical insights from financial research.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://cryptooptionpricing.streamlit.app/)

---

## 🚀 Key Features

- 📊 Download historical prices for crypto assets (e.g., SOL-USD from Yahoo Finance)
- 📈 Log-return computation, descriptive statistics, and **normality tests**
- 📉 Empirical return distribution fitting:
  - ✅ Standard Normal
  - ✅ Student’s t-distribution (with fitted degrees of freedom)
- 🧠 European option pricing using:
  - 🎲 **Monte Carlo simulation** with t-distribution
  - 📘 **Black-Scholes** formula with annualized volatility
- 📌 **ITM probability estimation** at expiry
- 📈 Visualization:
  - Simulated price paths
  - Terminal price distribution (rotated layout)
  - QQ plots (vs. normal and t-distribution)
- 📥 Export results and historical data as CSV
- 🧪 Modular structure: ready for multi-leg strategies (straddle, spreads, etc.)

---

## 🧠 Scientific Background

The app is motivated by recent research showing that **crypto return distributions** deviate significantly from normality, exhibiting **fat tails** better modeled by the **Student's t-distribution**. Relevant literature:

- Cassidy et al. (2010), *Pricing European Options with Log Student's t-distribution*  
  [arXiv:0906.4092](https://arxiv.org/abs/0906.4092)

- Zulfiqar & Gulzar (2021), *Bitcoin implied volatility smile analysis*  
  [SpringerOpen](https://jfin-swufe.springeropen.com/articles/10.1186/s40854-021-00280-y)

- Venter et al. (2021), *Bitcoin options with GARCH models*  
  [MDPI Journal of Risk and Financial Management](https://www.mdpi.com/1911-8074/14/6/261)

- Molin (2022), *Master thesis on crypto option pricing*  
  [Lund University](https://lup.lub.lu.se/student-papers/search/publication/9091224)

> See the [academic references wiki](https://github.com/gabrieledemarco/Crypto_Option_pricing/wiki/Academic-References) for more.

---

## 🧩 Project Structure

```
Crypto_Option_pricing/
├── app.py                          # Main Streamlit app
├── main.py                         # Optional CLI/test script
├── requirements.txt
├── option_analysis/
│   ├── __init__.py
│   ├── price_data_downloader.py   # Download historical prices (via Yahoo)
│   ├── return_analyzer.py         # Compute returns and fit t-distribution
│   ├── option_pricer.py           # Monte Carlo & Black-Scholes models
│   └── strike_sensitivity.py      # Multi-strike pricing sensitivity
```

---

## ▶️ How to Run

### 🌐 Run Online (Recommended)

No setup required — just click below to launch the app in the cloud:

👉 [Open App on Streamlit Cloud](https://cryptooptionpricing.streamlit.app/)

---

### 💻 Run Locally

1. Clone the repository:

```bash
git clone https://github.com/gabrieledemarco/Crypto_Option_pricing.git
cd Crypto_Option_pricing
```

2. (Optional) Create and activate a virtual environment:

```bash
python -m venv .venv
source .venv/bin/activate      # On macOS/Linux
.venv\Scripts ctivate         # On Windows
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

4. Launch the Streamlit app:

```bash
streamlit run app.py
```

The app will open at `https://cryptooptionpricing.streamlit.app/`.

---

## 📦 Requirements

- `streamlit`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `requests`
- `yfinance`

---

## ✅ Supported Use Cases

- Any asset available on Yahoo Finance (e.g., `SOL-USD`, `BTC-USD`, `ETH-USD`)
- User-customizable parameters:
  - ✅ Strike price
  - ✅ Expiry date
  - ✅ Risk-free rate

---

## 💡 Educational Use Case

This project is useful for:

- Finance and quantitative students
- Crypto and options traders
- Researchers interested in simulation-based pricing models

---

## 📬 Contact & Contribution

**This is an open educational project.**  
Feel free to fork, suggest improvements, or open issues!

- 📧 Email: [gabridemarco091@gmail.com](mailto:gabridemarco091@gmail.com)  
- 💼 LinkedIn: [Gabriele De Marco](https://www.linkedin.com/in/gabriele-de-marco-17a02ba7/)  
- ☕ Support: [Buy Me a Coffee](https://www.buymeacoffee.com/Gabridemarco95)

---

## 🧠 License

This project is released under the **MIT License**.  
Use it freely for study, customization, and experimentation.
