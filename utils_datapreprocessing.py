import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from scipy.stats import zscore
from pandas.plotting import scatter_matrix
from statsmodels.tsa.seasonal import seasonal_decompose

class StockAnalysis:
    def __init__(self, ticker, start_date, end_date, file_path, sheet_name, columns_to_merge):
        self.ticker = ticker
        self.start_date = start_date
        self.end_date = end_date
        self.file_path = file_path
        self.sheet_name = sheet_name
        self.columns_to_merge = columns_to_merge
        self.data = None

    def fetch_stock_data(self):
        df = yf.download(self.ticker, start=self.start_date, end=self.end_date)
        df.index = pd.to_datetime(df.index.date)
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'Date'}, inplace=True)
        df.set_index('Date', inplace=True)
        self.data = df

    def process_excel_data(self):
        stock_data = pd.read_excel(self.file_path, sheet_name=self.sheet_name)
        stock_data['Dates'] = pd.to_datetime(stock_data['Dates'])
        stock_data.set_index('Dates', inplace=True)

        desired_date_range = pd.date_range(start=self.start_date, end=self.end_date, freq='D')
        stock_data = stock_data.loc[stock_data.index.intersection(desired_date_range)]

        stock_data = stock_data[self.columns_to_merge]
        self.data = pd.merge(self.data, stock_data, left_index=True, right_index=True, how='left')

    def export_to_csv(self, filename):
        self.data.to_csv(filename, index=True)
        print(f"Data exported successfully as '{filename}'")

    def calculate_daily_returns(self):
        self.data['Daily_Return'] = self.data['Adj Close'].pct_change()
        self.data['cumulative_return'] = (1 + self.data['Daily_Return']).cumprod()

    def calculate_rolling_std(self, window=30):
        self.data['Rolling_Std'] = self.data['Daily_Return'].rolling(window=window).std()

    def detect_outliers(self):
        z_scores = self.data.apply(zscore)
        outliers = (z_scores > 3) | (z_scores < -3)
        outliers_summary = outliers.sum()
        print("Number of outliers per column:")
        print(outliers_summary)

    def plot_data(self):
        for column in self.data.columns:
            plt.figure(figsize=(12, 6))
            plt.plot(self.data.index, self.data[column])
            plt.title(column)
            plt.xlabel('Date')
            plt.ylabel(column)
            plt.show()

    def plot_data_by_column(self, column):
            plt.figure(figsize=(12, 6))
            plt.plot(self.data.index, self.data[column])
            plt.title(column)
            plt.xlabel('Date')
            plt.ylabel(column)
            plt.show()

    def scatter_matrix_plot(self):
        scatter_matrix(self.data, figsize=(15, 15), diagonal='kde')
        plt.suptitle('Scatter Matrix of Features')
        plt.show()

    def seasonal_decomposition(self, column, period=252):
        result = seasonal_decompose(self.data[column], model='additive', period=period)
        result.plot()
        plt.suptitle(f'Seasonal Decomposition of {column}', fontsize=10)
        plt.show()