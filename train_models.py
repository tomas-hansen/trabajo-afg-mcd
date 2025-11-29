# train_models.py
# Script to train all models from the notebook and save bundles

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
import os
import json
import time
import platform
from pathlib import Path
from joblib import dump
import shutil

# ML imports
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, mean_absolute_percentage_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.cross_decomposition import PLSRegression
import lightgbm as lgb
from lightgbm import LGBMRegressor
import tensorflow as tf
from tensorflow import keras

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None
sns.set_style("darkgrid")
sns.set(font_scale=1.1)

TARGET_COL = "% Iron Concentrate"


def reduce_mem_usage(df, verbose=True):
    numerics = ['int16', 'int32', 'int64', 'float16', 'float32', 'float64']
    start_mem = df.memory_usage().sum() / 1024**2
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose and start_mem > 0:
        print('Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)'.format(
            end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def timeseries_models_tracker_df():
    reg_models_scores_df = pd.DataFrame(
        columns=["model_name", "MAE", "RMSE", "MASE", "R2", "MAPE"])
    return reg_models_scores_df


def timeseries_report_model(y_test, model_preds, tracker_df="none", model_name="model_unknown", seasonality=1, naive=False):
    mae = round(mean_absolute_error(y_test, model_preds), 4)
    rmse = round(mean_squared_error(y_test, model_preds) ** 0.5, 4)
    mase = round(mean_absolute_scaled_error(
        y_test, model_preds, seasonality, naive), 4)
    r2 = round(r2_score(y_test, model_preds), 4)
    mape = round(mean_absolute_percentage_error(y_test, model_preds), 4)

    print("MAE: ", mae)
    print("RMSE :", rmse)
    print("MASE :", mase)
    print("R2 :", r2)
    print("MAPE :", mape)

    if isinstance(tracker_df, pd.core.frame.DataFrame):
        tracker_df.loc[tracker_df.shape[0]] = [
            model_name, mae, rmse, mase, r2, mape]
    else:
        pass


def mean_absolute_scaled_error(y_true, y_pred, seasonality=1, naive=False):
    y_true = np.array(y_true)
    mae = np.mean(np.abs(y_true - y_pred))
    if naive:
        mae_naive_no_season = np.mean(np.abs(y_true - y_pred))
    else:
        mae_naive_no_season = np.mean(
            np.abs(y_true[seasonality:] - y_true[:-seasonality]))
    return mae / mae_naive_no_season


def make_train_test_splits(windows, labels, test_split=0.2):
    split_size = int(len(windows) * (1-test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels


def plot_time_series(timesteps, values, start=0, end=None, label=None):
    sns.lineplot(x=timesteps[start:end], y=values[start:end], label=label)
    plt.xlabel("Date")
    plt.ylabel("% Iron Concentrate")
    if label:
        plt.legend(fontsize=14)
    plt.xticks(rotation=45)
    plt.grid(True)

# Data loading and preprocessing


def load_and_preprocess_data(csv_path="MiningProcess_Flotation_Plant_Database.csv"):
    df = pd.read_csv(csv_path, decimal=",", parse_dates=[
                     "date"], index_col="date")
    df = df.loc["2017-03-29 12:00:00":]
    df = reduce_mem_usage(df, False)

    # Fix duplicate row gap
    df_before = df.copy().loc[: "2017-04-10 00:00:00"]
    df_after = df.copy().loc["2017-04-10 01:00:00":]
    new_date = pd.to_datetime("2017-04-10 00:00:00")
    new_data = pd.DataFrame(
        df_before[-1:].values, index=[new_date], columns=df_before.columns)
    df = pd.concat([pd.concat([df_before, new_data]), df_after])

    # Build monotonic seconds index
    df.reset_index(allow_duplicates=True, inplace=True)
    df["duration"] = 20
    df.loc[0, "duration"] = 0
    df["duration"] = df["duration"].cumsum()
    df["Date_with_seconds"] = pd.Timestamp(
        "2017-03-29 12:00:00") + pd.to_timedelta(df["duration"], unit="s")
    df = df.set_index("Date_with_seconds").drop(columns=["index", "duration"])

    # Resample to hourly
    df_h = df.resample("H").first()
    df_h.index.names = ['Date']

    # Add cyclic features
    df_h["hora"] = df_h.index.hour
    df_h['sin_hora'] = np.sin(2 * np.pi * df_h["hora"]/24)
    df_h['cos_hora'] = np.cos(2 * np.pi * df_h["hora"]/24)
    df_h["dia_de_la_semana"] = df_h.index.day_of_week
    df_h['sin_dia_de_la_semana'] = np.sin(
        2 * np.pi * df_h["dia_de_la_semana"]/7)
    df_h['cos_dia_de_la_semana'] = np.cos(
        2 * np.pi * df_h["dia_de_la_semana"]/7)
    feat_eng_vars = ['sin_hora', 'cos_hora',
                     'sin_dia_de_la_semana', 'cos_dia_de_la_semana']
    feat_eng_df = df_h[feat_eng_vars]
    df_h = df_h.drop(["hora", "dia_de_la_semana"], axis=1)
    df_h = df_h.drop("% Silica Concentrate", axis=1)

    # Add 15min lags
    list_cols = [c for c in df.columns.to_list() if c not in [
        "% Silica Concentrate", "% Iron Concentrate"]]
    df_15 = df.resample("15min").first()
    df_15 = df_15.drop("% Silica Concentrate", axis=1)
    window_size = 3
    for col_name in list_cols:
        for i in range(window_size):
            df_15[f"{col_name} ({-15*(i+1)}mins)"] = df_15[col_name].shift(i + 1)

    df_h2 = df_15.resample("H").first()
    df_h2.index.names = ["Date"]

    # Add target lags
    for i in range(window_size):
        df_h2[f"{TARGET_COL} ({-i-1}h)"] = df_h2[TARGET_COL].shift(i + 1)

    df_h = df_h2.join(feat_eng_df, how="left")

    # Move target to first column
    cols = df_h.columns.to_list()
    if TARGET_COL in cols:
        cols.remove(TARGET_COL)
        cols.insert(0, TARGET_COL)
        df_h = df_h[cols]
    df_h = df_h.dropna().astype("float32")

    return df_h

# Function to save bundle


def save_bundle(bundle_dir, model, scaler, feature_names, model_name, extra_meta=None):
    bundle_dir = Path(bundle_dir)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Save model
    if hasattr(model, 'save'):  # Keras model
        model_path = bundle_dir / "model.keras"
        model.save(model_path)
    else:  # sklearn model
        model_path = bundle_dir / "model.joblib"
        dump(model, model_path)

    # Save scaler
    dump(scaler, bundle_dir / "scaler.joblib")

    # Save features
    with open(bundle_dir / "feature_names.json", "w", encoding="utf-8") as f:
        json.dump(feature_names, f, ensure_ascii=False, indent=2)

    # Save manifest
    meta = {
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "python_version": platform.python_version(),
        "model_type": model_name,
        "target": TARGET_COL,
        "scaler": "MinMaxScaler",
    }
    if extra_meta:
        meta.update(extra_meta)

    with open(bundle_dir / "manifest.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    # Zip
    zip_path = shutil.make_archive(str(bundle_dir), "zip", bundle_dir)
    print(f"Bundle saved: {zip_path}")

# Main training function


def main():
    print("Loading and preprocessing data...")
    df_h = load_and_preprocess_data()
    print(f"Data shape: {df_h.shape}")

    tracker = timeseries_models_tracker_df()

    # Prepare datasets
    df_h3 = df_h.copy()  # For LightGBM without target lags
    df_forecast = df_h.copy()  # Full

    # Common scaler
    scaler = MinMaxScaler()

    # Model 1: Naive
    print("\n=== Modelo Ingenuo ===")
    y_true = df_h[TARGET_COL]
    x_true = df_h[f"{TARGET_COL} (-1h)"]
    split_size = int(len(y_true) * 0.9)
    y_test = y_true[split_size:]
    x_test = x_true[split_size:]
    timeseries_report_model(
        y_test, x_test, tracker, model_name="Modelo Ingenuo", seasonality=1, naive=True)

    # Model 2: LightGBM
    print("\n=== LightGBM ===")
    X = df_h3.drop(TARGET_COL, axis=1)
    y = df_h3[TARGET_COL]
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
        X, y, test_split=0.1)

    # Simple params (from notebook)
    params = {"num_leaves": 31, "learning_rate": 0.05, "n_estimators": 300}
    model_lgbm = LGBMRegressor(
        objective="regression", random_state=123, **params)
    model_lgbm.fit(train_windows, train_labels)
    preds = model_lgbm.predict(test_windows)
    timeseries_report_model(test_labels, preds, tracker, model_name="LightGBM")

    # Save LightGBM bundle
    scaler_lgbm = MinMaxScaler()
    scaler_lgbm.fit(df_h3)
    df_scaled = pd.DataFrame(scaler_lgbm.transform(
        df_h3), index=df_h3.index, columns=df_h3.columns)
    X_scaled = df_scaled.drop(TARGET_COL, axis=1)
    save_bundle("lgbm_bundle", model_lgbm, scaler_lgbm,
                list(X_scaled.columns), "LightGBM")

    # Model 3: LSTM simple (few variables)
    print("\n=== LSTM (pocas variables) ===")
    df_lstm_simple = df_h.copy()
    for col in df_lstm_simple.columns:
        if "% Iron Conc" not in col:
            df_lstm_simple.drop(f"{col}", axis=1, inplace=True)

    scaler_lstm_simple = MinMaxScaler()
    scaler_lstm_simple.fit(df_lstm_simple)
    df_scaled = pd.DataFrame(scaler_lstm_simple.transform(
        df_lstm_simple), index=df_lstm_simple.index, columns=df_lstm_simple.columns)
    X = df_scaled.drop(TARGET_COL, axis=1).astype("float32")
    y = df_scaled[TARGET_COL].astype("float32")
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
        X, y, test_split=0.1)

    # Build and train LSTM
    tf.random.set_seed(42)
    inputs = tf.keras.layers.Input(shape=(train_windows.shape[1],))
    x = tf.keras.layers.Reshape((train_windows.shape[1], 1))(inputs)
    x = tf.keras.layers.LSTM(128, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="linear")(x)
    model_lstm_simple = tf.keras.Model(
        inputs, output, name="model_lstm_simple")
    model_lstm_simple.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005), metrics=["mae"])

    # Callbacks
    ckpt_dir = Path("weights_checkpoints/timeseries_model_lstm_simple")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "cp.keras"
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=20, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=str(
        checkpoint_path), monitor="loss", save_best_only=True, save_weights_only=False, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", patience=10, factor=0.2, verbose=1)

    model_lstm_simple.fit(x=train_windows, y=train_labels, epochs=120, verbose=0, batch_size=32, validation_data=(
        test_windows, test_labels), callbacks=[model_checkpoint, earlystopping, reduce_lr])

    best_model = keras.models.load_model(checkpoint_path, safe_mode=False)
    preds = pd.Series(best_model.predict(
        test_windows).reshape(-1), index=test_windows.index)
    y_real = pd.concat((test_labels, test_windows), axis=1)
    y_real = scaler_lstm_simple.inverse_transform(y_real)
    y_pred = pd.concat((preds, test_windows), axis=1)
    y_pred = scaler_lstm_simple.inverse_transform(y_pred)
    test_labels_inv = y_real[:, 0]
    preds_inv = y_pred[:, 0]
    timeseries_report_model(test_labels_inv, preds_inv,
                            tracker, model_name="LSTM (pocas variables)")

    # Save LSTM simple bundle
    save_bundle("lstm_simple_bundle", best_model, scaler_lstm_simple, list(
        df_lstm_simple.columns), "LSTM_simple", {"window_shape_train": list(train_windows.shape)})

    # Model 4: LSTM full
    print("\n=== LSTM full ===")
    scaler_lstm_full = MinMaxScaler()
    scaler_lstm_full.fit(df_forecast)
    df_scaled = pd.DataFrame(scaler_lstm_full.transform(
        df_forecast), index=df_forecast.index, columns=df_forecast.columns)
    X = df_scaled.drop(TARGET_COL, axis=1).astype("float32")
    y = df_scaled[TARGET_COL].astype("float32")
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
        X, y, test_split=0.1)

    # Same architecture
    inputs = tf.keras.layers.Input(shape=(train_windows.shape[1],))
    x = tf.keras.layers.Reshape((train_windows.shape[1], 1))(inputs)
    x = tf.keras.layers.LSTM(128, activation="relu")(x)
    x = tf.keras.layers.Dense(32, activation="relu")(x)
    x = tf.keras.layers.Dense(16, activation="relu")(x)
    output = tf.keras.layers.Dense(1, activation="linear")(x)
    model_lstm_full = tf.keras.Model(inputs, output, name="model_lstm_full")
    model_lstm_full.compile(loss="mae", optimizer=tf.keras.optimizers.Adam(
        learning_rate=0.0005), metrics=["mae"])

    ckpt_dir = Path("weights_checkpoints/timeseries_model_lstm_full")
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = ckpt_dir / "cp.keras"
    earlystopping = tf.keras.callbacks.EarlyStopping(
        monitor="loss", patience=20, restore_best_weights=True)
    model_checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath=str(
        checkpoint_path), monitor="loss", save_best_only=True, save_weights_only=False, verbose=1)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor="loss", patience=10, factor=0.2, verbose=1)

    model_lstm_full.fit(x=train_windows, y=train_labels, epochs=120, verbose=0, batch_size=32, validation_data=(
        test_windows, test_labels), callbacks=[model_checkpoint, earlystopping, reduce_lr])

    best_model = keras.models.load_model(checkpoint_path, safe_mode=False)
    preds = pd.Series(best_model.predict(
        test_windows).reshape(-1), index=test_windows.index)
    y_real = pd.concat((test_labels, test_windows), axis=1)
    y_real = scaler_lstm_full.inverse_transform(y_real)
    y_pred = pd.concat((preds, test_windows), axis=1)
    y_pred = scaler_lstm_full.inverse_transform(y_pred)
    test_labels_inv = y_real[:, 0]
    preds_inv = y_pred[:, 0]
    timeseries_report_model(test_labels_inv, preds_inv,
                            tracker, model_name="LSTM full")

    # Save LSTM full bundle
    save_bundle("lstm_full_bundle", best_model, scaler_lstm_full, list(
        df_forecast.columns), "LSTM_full", {"window_shape_train": list(train_windows.shape)})

    # Model 5: PLS
    print("\n=== PLS ===")
    X = df_forecast.drop(TARGET_COL, axis=1)
    y = df_forecast[TARGET_COL]
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
        X, y, test_split=0.1)

    # Find best n_components
    max_components = min(15, train_windows.shape[1])
    tscv = TimeSeriesSplit(n_splits=5)
    best_n_comp = 5  # Default, or implement search
    pls = PLSRegression(n_components=best_n_comp, scale=True)
    pls.fit(train_windows, train_labels)
    test_pred = pls.predict(test_windows).reshape(-1)
    preds = pd.Series(test_pred, index=test_windows.index)
    timeseries_report_model(test_labels, preds, tracker, model_name="PLS")

    # Save PLS bundle
    scaler_pls = MinMaxScaler()
    scaler_pls.fit(df_forecast)
    save_bundle("pls_bundle", pls, scaler_pls, list(
        df_forecast.columns), "PLS", {"n_components": best_n_comp})

    # Model 6: LightGBM multivariable
    print("\n=== LightGBM multivariable ===")
    X = df_forecast.drop(TARGET_COL, axis=1)
    y = df_forecast[TARGET_COL]
    train_windows, test_windows, train_labels, test_labels = make_train_test_splits(
        X, y, test_split=0.1)

    # Best params from search
    best_params = {"num_leaves": 31,
                   "learning_rate": 0.05, "n_estimators": 300}
    best_lgbm_full = lgb.LGBMRegressor(
        objective="regression", metric="rmse", **best_params)
    best_lgbm_full.fit(train_windows, train_labels)
    test_pred = best_lgbm_full.predict(test_windows)
    preds = pd.Series(test_pred, index=test_windows.index)
    timeseries_report_model(test_labels, preds, tracker,
                            model_name="LightGBM multivariable")

    # Save LightGBM full bundle
    scaler_lgbm_full = MinMaxScaler()
    scaler_lgbm_full.fit(df_forecast)
    save_bundle("lgbm_full_bundle", best_lgbm_full, scaler_lgbm_full,
                list(df_forecast.columns), "LightGBM_full")

    print("\n=== Resumen de Modelos ===")
    print(tracker)


if __name__ == "__main__":
    main()
