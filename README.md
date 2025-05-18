# 📈 Streamlit Option App – Analisi Opzioni Cripto con Monte Carlo & t-Student

Una web app interattiva per analizzare la convenienza economica delle opzioni call/put su asset cripto (es. SOL/USDT) utilizzando simulazioni Monte Carlo basate su distribuzioni **t-Student**, a confronto con il classico modello di **Black-Scholes**.

---

## 🚀 Funzionalità principali

- 📊 Download dati storici del sottostante (es. SOLUSDT) da Binance
- 📈 Calcolo log-return, statistica descrittiva, test di normalità
- 📉 Fit distribuzione empirica con:
  - ✅ Normale standard
  - ✅ t-Student (gradi di libertà stimati)
- 🧠 Pricing opzioni call/put con:
  - 🎲 Monte Carlo (t-student)
  - 📘 Black-Scholes
- 📌 Probabilità di profitto a scadenza (ITM)
- 📈 Visualizzazioni:
  - Nuvola di percorsi simulati
  - Distribuzione dei prezzi a scadenza (orientata orizzontalmente)
  - Confronto payoff netti attesi
- 📥 Esportazione risultati e prezzi storici in CSV
- 🧪 Pronto per estensione con strategie multi-opzione (es. straddle, vertical spread)

---

## 🧩 Struttura del progetto

```
option_analysis_package/
├── app.py                          # App principale Streamlit
├── option_analysis/
│   ├── __init__.py
│   ├── price_data_downloader.py   # Scarica dati da Binance
│   ├── return_analyzer.py         # Calcola log-return e stima t-student
│   ├── option_pricer.py           # Monte Carlo e Black-Scholes
│   └── strike_sensitivity.py      # Analisi su strike multipli
├── main.py                        # Script test/debug
```

---

## ▶️ Come eseguirlo

1. Installa i requisiti:

```bash
pip install -r requirements.txt
```

2. Avvia l'app Streamlit:

```bash
streamlit run app.py
```

---

## 📦 Requisiti

- `streamlit`
- `numpy`
- `pandas`
- `scipy`
- `matplotlib`
- `seaborn`
- `requests`

---

## 📌 Esempi supportati

- ✅ SOLUSDT (Binance)
- ✅ Strike price personalizzabile
- ✅ Scadenza selezionabile
- ✅ Tasso risk-free configurabile
- (espandibile facilmente a ETH, BTC, ecc.)

---

## 📬 Contatti e contributi

Progetto didattico / sperimentale.  
Per suggerimenti o richieste, contattami direttamente.

---

## 🧠 Licenza

MIT – libero utilizzo per fini di analisi, studio e personalizzazione.
