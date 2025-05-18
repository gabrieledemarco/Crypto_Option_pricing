import pandas as pd
import yfinance as yf

class PriceDataDownloader:
    def __init__(self, symbol="SOL-USD", interval="1d", period="4y"):
        self.symbol = symbol
        self.interval = interval
        self.period = period

    def fetch_binance_data(self):
        # Scarica i dati da Yahoo Finance
        df = yf.download(
            tickers=self.symbol,
            period=self.period,
            interval=self.interval,
            progress=False,
            auto_adjust=True
        )

        # Pulisce e formatta come richiesto dal tuo progetto
        df = df[["Close"]].rename(columns={"Close": "close"})
        df.index.name = "timestamp"
        df.dropna(inplace=True)

        return df
