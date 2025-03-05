import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.utils import plot_model
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.preprocessing.sequence import TimeseriesGenerator
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from tensorflow.keras.layers import LSTM, Dense, Dropout, Input
from sklearn.model_selection import train_test_split
from joblib import dump, load
from keras_tuner.tuners import RandomSearch, Hyperband, BayesianOptimization

import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import os
import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, auc, roc_curve
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class LSTMModel:
    def __init__(self, seqlen=60, numfeat=3):
        self.seqlen = seqlen
        self.numfeat = numfeat
        self.model_path = None
        self.model = None
        self.scaler = StandardScaler()
        self.results_path = Path('results', 'lstm_time_series')
        if not self.results_path.exists():
            self.results_path.mkdir(parents=True)

    def load_data(self):
        self.X_pca_transformed = load(os.path.join("pca_output", "X_pca_transformed.pkl"))
        print(f"Loaded PCA-transformed data: {self.X_pca_transformed.shape[1]}")
        self.numfeat=self.X_pca_transformed.shape[1]
        self.y_target = load(os.path.join("pca_output", "y_target.pkl"))
        Xtrain, Xtest, ytrain, ytest = train_test_split(self.X_pca_transformed, self.y_target, test_size=0.20, shuffle=False)
        self.X_train_pca = Xtrain
        self.X_test_pca = Xtest
        self.y_train = ytrain
        self.y_test = ytest

        # Check and process `X_train_pca`
        if isinstance(self.X_train_pca, pd.Series):
            self.X_train_pca = self.X_train_pca.to_frame()  # Convert to DataFrame if it's a Series
        if isinstance(self.X_train_pca, pd.DataFrame):
            self.X_train_pca = self.X_train_pca.to_numpy()  # Convert to numpy array
        if self.X_train_pca.dtype != np.float32:
            self.X_train_pca = self.X_train_pca.astype(np.float32)  # Convert to float32
            print("\nConverted X_train_pca to np.float32")

        # Check and process `y_train`
        if isinstance(self.y_train, pd.Series) or isinstance(self.y_train, pd.DataFrame):
            self.y_train = self.y_train.to_numpy()  # Convert to numpy array
        if self.y_train.dtype != np.int32:
            self.y_train = self.y_train.astype(np.int32)
            print("Converted y_train to np.int32")

    def preprocess_data(self):
        self.X_train_scaled = self.scaler.fit_transform(self.X_train_pca)
        self.X_test_scaled = self.scaler.transform(self.X_test_pca)
        self.train_generator = TimeseriesGenerator(self.X_train_scaled, self.y_train, length=self.seqlen)
        self.test_generator = TimeseriesGenerator(self.X_test_scaled, self.y_test, length=self.seqlen)

    def create_model(self, hu=256):
        tf.keras.backend.clear_session()
        model = Sequential()
        model.add(Input(shape=(self.seqlen, self.numfeat), name='InputLayer'))
        model.add(LSTM(units=hu*2, activation='elu', return_sequences=True, name='LSTM1'))
        model.add(Dropout(0.4, name='Dropout1'))
        model.add(LSTM(units=hu, activation='elu', return_sequences=True, name='LSTM2'))
        model.add(Dropout(0.4, name='Dropout2'))
        model.add(LSTM(units=hu, activation='elu', return_sequences=False, name='LSTM3'))
        model.add(Dense(units=1, activation='sigmoid', name='Output'))
        opt = Adam(learning_rate=0.001, epsilon=1e-08)
        model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=['accuracy', Precision(), Recall()])
        return model

    def train_model(self, hu=10, epochs=250):
        self.model = self.create_model(hu)
        logdir = os.path.join("./tensorboard/logs", dt.datetime.now().strftime("%Y%m%d-%H%M%S"))
        self.model_path = (self.results_path / 'model.keras').as_posix()
        my_callbacks = [
            EarlyStopping(patience=20, monitor='loss', mode='min', verbose=1, restore_best_weights=True),
            ModelCheckpoint(filepath=self.model_path, verbose=1, monitor='loss', save_best_only=True),
            TensorBoard(log_dir=logdir, histogram_freq=1)
        ]
        self.model.fit(self.train_generator, epochs=epochs, verbose=2, callbacks=my_callbacks, shuffle=False)

    def evaluate_model(self):
        y_pred_prob = self.model.predict(self.test_generator, verbose=0).ravel()
        y_pred = (y_pred_prob > 0.5).astype("int32")
        y_test_trimmed = self.y_test[self.seqlen:]
        acc = accuracy_score(y_test_trimmed, y_pred)
        prec = precision_score(y_test_trimmed, y_pred)
        rec = recall_score(y_test_trimmed, y_pred)
        f1 = f1_score(y_test_trimmed, y_pred)
        cm = confusion_matrix(y_test_trimmed, y_pred)
        fpr, tpr, thresholds = roc_curve(y_test_trimmed, y_pred_prob)
        roc_auc = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver Operating Characteristic (ROC) Curve')
        plt.legend(loc="lower right")
        plt.grid()
        plt.tight_layout()
        plt.show(block=True)
        return acc, prec, rec, f1, cm, roc_auc

    def plot_model(self):
        plot_model(self.model, to_file='model.png', show_shapes=True, show_layer_names=True)

    def save_model(self):
        self.model.save('optimized_lstm_model.keras')
        print("Optimized LSTM model saved.")

    def load_model(self):
        self.model = load_model(self.model_path)
        print("Model loaded successfully.")

    def check_feature_mismatch(self, X):
        if X.shape[1] != self.numfeat:
            print(f"Feature mismatch: Data has {X.shape[1]} features, but model expects {self.numfeat}.")
        else:
            print("Feature dimensions match!")
        if len(X) < self.seqlen:
            print(f"Insufficient data: Need at least {self.seqlen} samples for the sequence length.")
        else:
            print("Sequence length is sufficient.")

    def hyperparameter_tuning(self, tuner_type='random', max_trials=20, epochs=50):
        def build_model(hp):
            tf.keras.backend.clear_session()
            model = Sequential()
            model.add(Input(shape=(self.seqlen, self.numfeat), name='InputLayer'))
            hp_units1 = hp.Int('units1', min_value=4, max_value=32, step=4)
            hp_units2 = hp.Int('units2', min_value=4, max_value=32, step=4)
            hp_units3 = hp.Int('units3', min_value=4, max_value=32, step=4)
            hp_dropout1 = hp.Float('Dropout_rate1', min_value=0, max_value=0.5, step=0.1)
            hp_dropout2 = hp.Float('Dropout_rate2', min_value=0, max_value=0.5, step=0.1)
            hp_activation1 = hp.Choice('activation1', values=['relu', 'elu', 'tanh'])
            hp_activation2 = hp.Choice('activation2', values=['relu', 'elu', 'tanh'])
            hp_activation3 = hp.Choice('activation3', values=['relu', 'elu', 'tanh'])
            hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])
            hp_batch_size = hp.Choice('batch_size', [32, 64, 128])
            model.add(LSTM(hp_units1, input_shape=(self.seqlen, self.numfeat), activation=hp_activation1, return_sequences=True))
            model.add(Dropout(hp_dropout1))
            model.add(LSTM(hp_units2, activation=hp_activation2, return_sequences=True))
            model.add(Dropout(hp_dropout2))
            model.add(LSTM(hp_units3, activation=hp_activation3, return_sequences=False))
            model.add(Dense(units=1, activation='sigmoid'))
            opt = Adam(learning_rate=hp_learning_rate)
            model.compile(optimizer=opt, loss=BinaryCrossentropy(), metrics=['accuracy', Precision(), Recall()])
            return model

        if tuner_type == 'random':
            tuner = RandomSearch(build_model, objective='val_accuracy', max_trials=max_trials, directory='hyper_tuning', project_name='lstm_tuning', overwrite=True)
        elif tuner_type == 'hyperband':
            tuner = Hyperband(build_model, objective='val_accuracy', max_epochs=5, hyperband_iterations=15, directory='hyper_tuning', project_name="hbtrail", overwrite=True)
        elif tuner_type == 'bayesian':
            tuner = BayesianOptimization(build_model, objective="val_accuracy", max_trials=max_trials, num_initial_points=2, directory='hyper_tuning', project_name="botrial", overwrite=True)
        else:
            raise ValueError("Invalid tuner type. Choose 'random', 'hyperband', or 'bayesian'.")

        tuner.search(self.train_generator, validation_data=self.test_generator, epochs=epochs, callbacks=[EarlyStopping(patience=5, monitor='loss', mode='min', verbose=1, restore_best_weights=True), TensorBoard(log_dir="./tensorboard/logs")], shuffle=False)

        best_hp = tuner.get_best_hyperparameters()[0]
        best_trial = tuner.oracle.get_best_trials()[0]
        best_val_accuracy = best_trial.metrics.get_last_value("val_accuracy")
        print("Best Validation Accuracy:", best_val_accuracy)
        print("Best Hyperparameters:", best_hp.values)

        self.model = tuner.hypermodel.build(best_hp)
        self.model.fit(self.train_generator, validation_data=self.test_generator, epochs=epochs, callbacks=[EarlyStopping(patience=20, monitor='loss', mode='min', verbose=1, restore_best_weights=True), ModelCheckpoint(filepath="optimized_lstm_model.keras", verbose=1, monitor='loss', save_best_only=True), TensorBoard(log_dir="./tensorboard/logs")], shuffle=False)
        self.save_model()

    def evaluate_and_visualize(self):
        y_pred_prob_final = self.model.predict(self.test_generator, verbose=0).ravel()
        y_pred_final = (y_pred_prob_final > 0.5).astype("int32")
        y_test_trimmed_final = self.y_test[self.seqlen:]
        acc_final = accuracy_score(y_test_trimmed_final, y_pred_final)
        prec_final = precision_score(y_test_trimmed_final, y_pred_final)
        rec_final = recall_score(y_test_trimmed_final, y_pred_final)
        f1_final = f1_score(y_test_trimmed_final, y_pred_final)
        cm_final = confusion_matrix(y_test_trimmed_final, y_pred_final)
        fpr, tpr, thresholds = roc_curve(y_test_trimmed_final, y_pred_prob_final)
        roc_auc_final = auc(fpr, tpr)
        plt.figure()
        plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc_final:.2f})')
        plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
        plt.title('Receiver Operating Characteristic - Post Hyperparamater Tuning')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.legend(loc="lower right")
        plt.tight_layout()
        plt.show(block=True)
        return acc_final, prec_final, rec_final, f1_final, cm_final, roc_auc_final

    def save_predictions(self, filepath="aligned_data.csv"):
        y_pred = np.where(self.model.predict(self.test_generator, verbose=False) > 0.5, 1, 0)
        lstm_tradingstrat_data = pd.read_csv('features_df_prefeatselect.csv', index_col='Date', parse_dates=True)
        aligned_data = lstm_tradingstrat_data.iloc[-(len(self.y_test) - self.seqlen):].copy()
        aligned_data['y_pred'] = y_pred
        aligned_data.to_csv(filepath, index=True)
        print(f"Predictions saved to {filepath}")

    def run(self, hu=10, epochs=250, tuner_type='random', max_trials=20, tuning_epochs=50):
        self.load_data()
        self.preprocess_data()
        self.train_model(hu, epochs)
        self.evaluate_model()
        self.hyperparameter_tuning(tuner_type, max_trials, tuning_epochs)
        self.evaluate_and_visualize()
        self.save_predictions()