import yfinance as yf
import pandas as pd
import numpy as np


class CrossAssetFeatureEngineering:
    def __init__(self, tickers, start_date, end_date):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.close_prices = None
        self.features = None

    def download_data(self):
        # Download only 'Close' prices for all tickers
        self.close_prices = pd.DataFrame({
            ticker: yf.download(ticker, start=self.start_date, end=self.end_date)["Close"] for ticker in self.tickers
        }).dropna()

    def align_data(self):
        # Initialize L'Oréal's data
        loreal_close = self.close_prices["OR.PA"]

        # Align all securities to match L'Oréal's index
        self.close_prices = self.close_prices.reindex(loreal_close.index).dropna()

    def calculate_features(self):
        # Create an empty DataFrame to store calculated features
        self.features = pd.DataFrame(index=self.close_prices.index)
        loreal_close = self.close_prices["OR.PA"]

        # Calculate log returns and lagged log returns for other securities
        for ticker in self.tickers:
            if ticker != "OR.PA":
                # Log returns
                self.features[f"Log_Returns_{ticker}"] = np.log(self.close_prices[ticker] / self.close_prices[ticker].shift(1))

                # Lagged log returns
                for lag in [1, 5, 7, 14]:
                    self.features[f"Lagged_Log_Returns_{ticker}_{lag}"] = self.features[f"Log_Returns_{ticker}"].shift(lag)

                # Rolling correlation with L'Oréal
                self.features[f"Rolling_Corr_{ticker}"] = loreal_close.rolling(window=30).corr(self.close_prices[ticker])

                # Price ratio and price spread with L'Oréal
                self.features[f"Price_Ratio_{ticker}"] = loreal_close / self.close_prices[ticker]
                self.features[f"Price_Spread_{ticker}"] = loreal_close - self.close_prices[ticker]

    def save_features(self, filename):
        # Save the final dataframe to CSV for inspection
        self.features.to_csv(filename, index=True)
        print(f"Features saved to {filename}")

    def display_nan_counts(self):
        nan_counts = self.features.isna().sum()
        print("NaN Counts:")
        print(nan_counts)

    def display_sample(self):
        print("Head of Features:")
        print(self.features.head())
        print("Tail of Features:")
        print(self.features.tail())