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
            "symbol": self.symbol,
            "interval": self.interval,
            "limit": 1000
        }
        data = []
        start_time = pd.Timestamp.now() - pd.Timedelta(days=self.days)
        for _ in range((self.days // 1000) + 1):
            params["startTime"] = int(start_time.timestamp() * 1000)
            res = requests.get(url, params=params).json()
            data.extend(res)
            start_time += pd.Timedelta(days=1000)
        df = pd.DataFrame(data, columns=["timestamp", "open", "high", "low", "close", "volume",
                                         "close_time", "quote_asset_volume", "number_of_trades",
                                         "taker_buy_base", "taker_buy_quote", "ignore"])
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms")
        df["close"] = df["close"].astype(float)
        df.set_index("timestamp", inplace=True)
        return df[["close"]]
