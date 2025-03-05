import numpy as np
import pandas as pd
import pandas_ta as ta
from sklearn.preprocessing import StandardScaler
from scipy.spatial.distance import cdist
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import VarianceThreshold
from sklearn.decomposition import PCA
from statsmodels.tsa.seasonal import seasonal_decompose
import os
import seaborn as sns
from scipy.stats import skew, kurtosis
from sklearn.preprocessing import PowerTransformer
import matplotlib.pyplot as plt
from joblib import dump, load

class FeatureEngineering:
    def __init__(self, file_path):
        self.file_path = file_path
        self.data = None
        self.scaled_data = None
        self.features_df = None
        self.feature_importances = None
        self.X_final = None
        self.y_final = None

    def load_data(self):
        self.data = pd.read_csv(self.file_path, index_col='Date', parse_dates=True)
        self.features_df = self.data.copy()
        print(f"Data loaded successfully with shape {self.features_df.shape}")

    def add_technical_indicators(self):
        self.features_df.ta.study(ta.AllStrategy)
        print("Added technical indicators")

    def add_valuation_ratios(self):
        self.features_df['PE_to_Dividend'] = self.features_df['PriceToEarnings'] / self.features_df['DividendYield']
        self.features_df['PE_to_PB'] = self.features_df['PriceToEarnings'] / self.features_df['PriceToBook']
        self.features_df['PC_to_E'] = self.features_df['PriceToCash'] / self.features_df['PriceToEarnings']
        self.features_df['CWM'] = (self.features_df['PriceToEarnings'] / self.features_df['PriceToBook']) * (self.features_df['PriceToCash'] / self.features_df['PriceToEarnings'])
        print("Added valuation ratios and financial health metrics")

    def normalize_data(self):
        scaler = StandardScaler()
        self.scaled_data = scaler.fit_transform(self.features_df[['Close']])
        print("'Close' Data normalized for Recurrence Properties")

    def recurrence_plot(self, data, threshold=0.1):
        distance_matrix = cdist(data, data, 'euclidean')
        return (distance_matrix < threshold).astype(int)

    def calculate_entropy(self, rp):
        prob_matrix = np.mean(rp, axis=0)
        prob_nonzero = prob_matrix[prob_matrix > 0]
        entropy = -np.sum(prob_nonzero * np.log(prob_nonzero))
        return entropy

    def add_recurrence_features(self, window_size=30, threshold=0.1):
        recurrence_percentages = []
        recurrence_entropies = []

        for i in range(window_size, len(self.features_df)):
            window_data = self.scaled_data[i-window_size:i]
            rp = self.recurrence_plot(window_data, threshold)
            recurrence_percentage = np.sum(rp) / rp.size * 100
            recurrence_percentages.append(recurrence_percentage)
            recurrence_entropy = self.calculate_entropy(rp)
            recurrence_entropies.append(recurrence_entropy)

        self.features_df['recurrence_percentage'] = [None] * window_size + recurrence_percentages
        self.features_df['recurrence_entropy'] = [None] * window_size + recurrence_entropies
        print("Added recurrence features")

    def add_lagged_features(self, lags):
        self.features_df['daily_return'] = self.features_df['Adj Close'].pct_change()
        self.features_df['log_return'] = np.log(self.features_df['Adj Close'] / self.features_df['Adj Close'].shift(1))
        for lag in lags:
            self.features_df[f"Lagged_Return_{lag}"] = self.features_df['daily_return'].shift(lag)
            self.features_df[f"Lagged_Log_Return_{lag}"] = self.features_df['log_return'].shift(lag)
            self.features_df[f"Rolling_Volatility_{lag}"] = self.features_df['daily_return'].rolling(window=lag).std()
        print("Added lagged features")

    def decompose_time_series(self, column, period=252):
        result = seasonal_decompose(self.features_df[column], model='additive', period=period)
        return {
            'trend': result.trend,
            'seasonal': result.seasonal,
            'residual': result.resid
        }

    def add_decomposed_features(self):
        adj_close_decomp = self.decompose_time_series('Adj Close')
        volume_decomp = self.decompose_time_series('Volume')
        self.features_df['AdjClose_Trend'] = adj_close_decomp['trend']
        self.features_df['AdjClose_Seasonal'] = adj_close_decomp['seasonal']
        self.features_df['AdjClose_Residual'] = adj_close_decomp['residual']
        self.features_df['Volume_Trend'] = volume_decomp['trend']
        self.features_df['Volume_Seasonal'] = volume_decomp['seasonal']
        self.features_df['Volume_Residual'] = volume_decomp['residual']
        print("Added decomposed features")

    def merge_cross_asset_features(self, cross_asset_file):
        cross_asset_data = pd.read_csv(cross_asset_file, index_col='Date', parse_dates=True)
        self.features_df = pd.merge(self.features_df, cross_asset_data, left_index=True, right_index=True, how='left')
        print("Merged cross-asset features")

    def drop_unwanted_columns(self, columns):
        self.features_df.drop(columns, axis=1, inplace=True)
        print("Dropped unwanted columns")

    def handle_missing_values(self):
        self.features_df.dropna(inplace=True)
        print("Handled missing values")

    def shift_label(self):
        self.features_df['y'] = np.where(self.features_df['daily_return'].shift(-1) > 0, 1, 0)
        print("Shifted label for next day's movement")

    def save_features(self, output_path):
        self.features_df.to_csv(output_path, index=True)
        print(f"Features saved to {output_path}")

    def feature_selection(self):
         # Step 1: Separate Features and Target (y)
        X = self.features_df.drop(columns=['y'])  # Features
        y = self.features_df['y']  # Target
        
        print(f"Initial number of features: {X.shape[1]}")
        
        # Step 2: Assess Correlation Matrix Between Features
        # Compute correlation matrix for the features
        corr_matrix = X.corr().abs()
        
        
        # Visualize the correlation matrix
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', annot=False, fmt='.2f', 
                    cbar=True, square=True, linewidths=0.5)
        plt.title('Feature Correlation Matrix')
        plt.show()
        
        
        # Identify feature pairs with high correlation (> 0.9)
        high_corr_pairs = [
            (col, row) for col in corr_matrix.columns for row in corr_matrix.index 
            if col != row and corr_matrix.loc[row, col] > 0.9
        ]
        
        print(f"Number of highly correlated feature pairs: {len(high_corr_pairs)}")
        
        # Step 3: Drop One Feature from Each High-Correlation Pair Based on Importance
        # Fit a Random Forest model for feature importance
        rf = RandomForestClassifier(random_state=42)
        rf.fit(X, y)
        feature_importances = pd.Series(rf.feature_importances_, index=X.columns)
        
        # Track features to drop
        features_to_drop = set()
        
        for col1, col2 in high_corr_pairs:
            if col1 not in features_to_drop and col2 not in features_to_drop:
                # Compare feature importances
                if feature_importances[col1] > feature_importances[col2]:
                    features_to_drop.add(col2)
                else:
                    features_to_drop.add(col1)
        
        print(f"Features dropped due to high correlation and low importance: {list(features_to_drop)}")
        
        # Drop the selected features
        X_after_corr_filter = X.drop(columns=list(features_to_drop))
        
        print(f"Number of features after high correlation filtering: {X_after_corr_filter.shape[1]}")
        
        
        # Step 1: PCA for Dimensionality Reduction
        pca = PCA(n_components=0.95)
        X_pca = pca.fit_transform(X_after_corr_filter)
        print(f"Number of features before PCA: {X_after_corr_filter.shape[1]}")
        print(f"Number of features after PCA: {X_pca.shape[1]}")
        
        # Step 2: Loading Matrix
        loading_matrix = pd.DataFrame(
            pca.components_.T,
            index=X_after_corr_filter.columns,
            columns=[f'PC{i+1}' for i in range(pca.n_components_)]
        )
        
        # Step 3: Top Contributing Features
        top_features_per_pc = {}
        for component in loading_matrix.columns:
            sorted_features = loading_matrix[component].abs().sort_values(ascending=False)
            top_features = sorted_features.head(5)
            top_features_per_pc[component] = top_features.index.tolist()
            print(f"\nTop contributing features for {component}:")
            print(top_features)
        
        # Step 4: Skewness and Kurtosis Before Transformation
        X_pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
        skewness = X_pca_df.apply(skew)
        kurtosis_values = X_pca_df.apply(kurtosis)
        print("\nSkewness and Kurtosis of Principal Components:")
        print(pd.DataFrame({'Skewness': skewness, 'Kurtosis': kurtosis_values}))
        
        # Step 5: Yeo-Johnson Transformation
        pt = PowerTransformer(method='yeo-johnson')
        X_pca_transformed = pt.fit_transform(X_pca_df)
        X_pca_transformed_df = pd.DataFrame(X_pca_transformed, columns=X_pca_df.columns)
        
        # Step 6: Skewness and Kurtosis After Transformation
        skewness_after = X_pca_transformed_df.apply(skew)
        kurtosis_after = X_pca_transformed_df.apply(kurtosis)
        print("\nSkewness after transformation:")
        print(skewness_after)
        print("\nKurtosis after transformation:")
        print(kurtosis_after)
        
        # Step 7: Visualization
        for column in X_pca_transformed_df.columns:
            plt.figure(figsize=(6, 4))
            plt.hist(X_pca_transformed_df[column], bins=30, alpha=0.7, label=column, edgecolor='black')
            plt.title(f'Distribution of {column}')
            plt.xlabel('Value')
            plt.ylabel('Frequency')
            plt.legend()
            plt.show()

        self.X_final = pd.DataFrame(X_pca_transformed, index=X_after_corr_filter.index, columns=[f'PC{i+1}' for i in range(X_pca_transformed.shape[1])])
        self.y_final = y  # Assuming 'y' is the target variable already defined
        
    def save_transformed_data(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        dump(self.X_final, os.path.join(output_dir, "X_pca_transformed.pkl"))
        dump(self.y_final, os.path.join(output_dir, "y_target.pkl"))
        print(f"PCA-transformed data and target saved in {output_dir}")

    def run(self, cross_asset_file, unwanted_columns, output_path_features, output_path_transformed):
        self.load_data()
        self.add_technical_indicators()
        self.add_valuation_ratios()
        self.normalize_data()
        self.add_recurrence_features()
        self.add_decomposed_features()
        self.merge_cross_asset_features(cross_asset_file)
        self.add_lagged_features([1, 5, 14, 28, 50])
        self.drop_unwanted_columns(unwanted_columns)
        self.handle_missing_values()
        self.shift_label()
        self.save_features(output_path_features)
        self.feature_selection()
        self.save_transformed_data(output_path_transformed)
