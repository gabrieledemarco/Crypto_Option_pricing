import pandas as pd
import requests

class PriceDataDownloader:
    def __init__(self, symbol="SOLUSDT", interval="1d", days=1460):
        self.symbol = symbol
        self.interval = interval
        self.days = days

    def fetch_binance_data(self):
        url = "https://api.binance.com/api/v3/klines"
        params = {
            "symbol": self.symbol.upper(),
            "interval": self.interval,
            "limit": 1000
        }

        data = []
        start_time = pd.Timestamp.now() - pd.Timedelta(days=self.days)

        for _ in range((self.days // 1000) + 1):
            params["startTime"] = int(start_time.timestamp() * 1000)

            try:
                response = requests.get(url, params=params)
                response.raise_for_status()
                res = response.json()

                # Sanity check: Binance returns list of lists (candles)
                if not res or not isinstance(res, list) or not all(isinstance(candle, list) and len(candle) == 12 for candle in res):
                    raise ValueError("Formato dati inatteso dalla Binance API")

                data.extend(res)
                start_time += pd.Timedelta(days=1000)

            except Exception as e:
                print(f"Errore durante il download dei dati: {e}")
                break  # o `continue` se vuoi provare a saltare l'intervallo

        if not data:
            raise ValueError("Nessun dato scaricato")

        # Costruzione DataFrame
        df = pd.DataFrame(data, columns=[
            "timestamp", "open", "high", "low", "close", "volume",
            "close_time", "quote_asset_volume", "number_of_trades",
            "taker_buy_base", "taker_buy_quote", "ignore"
        ])

        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close"] = df["close"].astype(float)
        df.set_index("timestamp", inplace=True)

        return df[["close"]]
