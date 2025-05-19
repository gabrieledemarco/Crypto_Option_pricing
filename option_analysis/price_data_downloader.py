"""
Modulo per il download dei prezzi storici da Yahoo Finance.
"""

import yfinance as yf


class PriceDataDownloader:
    """Classe per scaricare i dati di prezzo da Yahoo Finance."""

    def __init__(self, symbol="SOL-USD", interval="1d", period="4y"):
        """Inizializza il downloader con simbolo, intervallo e periodo."""
        self.symbol = symbol
        self.interval = interval
        self.period = period

    def fetch_data(self):
        """Scarica i dati da Yahoo Finance e restituisce un DataFrame pulito."""
        try:
            df = yf.download(
                tickers=self.symbol,
                period=self.period,
                interval=self.interval,
                progress=False,
                auto_adjust=True
            )
        except Exception as e:
            raise ConnectionError(f"Errore nel download dati da Yahoo Finance: {e}") from e

        if "Close" not in df.columns:
            raise ValueError("Dati non validi o simbolo errato.")

        df = df[["Close"]].rename(columns={"Close": "close"})
        df.index.name = "timestamp"
        df.dropna(inplace=True)

        return df
