import numpy as np
import pandas as pd
import quantstats as qs
import matplotlib.pyplot as plt

class LSTMTradingStrategyBacktest:
    def __init__(self, file_path, commission=0.002, starting_cash=10000, position_size=1.0):
        self.file_path = file_path
        self.commission = commission
        self.starting_cash = starting_cash
        self.position_size = position_size
        self.data = None
        self._load_data()

    def _load_data(self):
        """Load data from the CSV file and prepare the DataFrame."""
        self.data = pd.read_csv(self.file_path, index_col='Date', parse_dates=True)
        print("Data loaded successfully.")

    def generate_signals(self):
        """Generate LSTM trading signals."""
        self.data['lstm_signal'] = self.data['y_pred']

    def calculate_strategy_returns(self):
        """Calculate returns based on the LSTM signals."""
        self.data['lstm_strategy_return'] = (
            self.data['lstm_signal'] * self.data['Lagged_Return_1']
        )

    def adjust_for_commission(self):
        """Adjust strategy returns for trading commissions."""
        self.data['lstm_signal_shifted'] = self.data['lstm_signal'].shift(1, fill_value=0)
        self.data['trade_commission'] = (
            (self.data['lstm_signal'] != self.data['lstm_signal_shifted']) * self.commission
        )
        self.data['adjusted_momentum_strategy_return'] = (
            (
                self.data['lstm_signal'] * self.data['Lagged_Return_1'] -
                self.data['trade_commission']
            ) * self.position_size
        )

    def calculate_cumulative_returns(self):
        """Calculate cumulative returns for the strategy and benchmark."""
        self.data['cumulative_strategy_return'] = (
            1 + self.data['adjusted_momentum_strategy_return']
        ).cumprod()
        self.data['cumulative_benchmark_return'] = (
            1 + self.data['Lagged_Return_1']
        ).cumprod()

    def calculate_excess_returns(self):
        """Calculate excess returns over the benchmark."""
        self.data['excess_return'] = (
            self.data['adjusted_momentum_strategy_return'] - self.data['Lagged_Return_1']
        )

    def plot_cumulative_returns(self):
        """Plot cumulative returns for the strategy and benchmark."""
        self.data[['cumulative_strategy_return', 'cumulative_benchmark_return']].plot(title="Cumulative Returns")
        plt.xlabel("Date")
        plt.ylabel("Cumulative Return")
        plt.legend(["Strategy", "Benchmark"])
        plt.show()

    def generate_quantstats_report(self, output_path=None):
        """Generate QuantStats performance report."""
        strategy_returns = self.data['adjusted_momentum_strategy_return'].fillna(0)
        benchmark_returns = self.data['Lagged_Return_1'].fillna(0)

        if output_path:
            qs.reports.html(
                strategy_returns,
                benchmark=benchmark_returns,
                title="LSTM Trading Strategy Performance",
                output=output_path
            )
        else:
            qs.reports.full(strategy_returns, benchmark=benchmark_returns)

    def run_backtest(self, generate_report=True, report_path="lstm_strategy_performance.html"):
        """Run the entire backtest process."""
        self.generate_signals()
        self.calculate_strategy_returns()
        self.adjust_for_commission()
        self.calculate_cumulative_returns()
        self.calculate_excess_returns()
        self.plot_cumulative_returns()

        if generate_report:
            self.generate_quantstats_report(output_path=report_path)