# evaluate_models.py
# Script to load all trained models and evaluate their performance on test set

import numpy as np
import pandas as pd
import warnings
from utils import (
    preprocess_csv_to_hourly_features,
    load_xgb_bundle, load_lstm_bundle, load_lgbm_bundle, load_pls_bundle,
    timeseries_models_tracker_df, timeseries_report_model, mean_absolute_scaled_error
)

warnings.filterwarnings("ignore")

TARGET_COL = "% Iron Concentrate"


def load_and_preprocess_data(csv_path="MiningProcess_Flotation_Plant_Database.csv"):
    df = pd.read_csv(csv_path, decimal=",", parse_dates=[
                     "date"], index_col="date")
    df = df.loc["2017-03-29 12:00:00":]
    df = df.dropna()  # Ensure no NaN

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
            df_15[f"{col_name} ({-15 * (i + 1)}mins)"] = df_15[col_name].shift(i + 1)

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


def make_train_test_splits(windows, labels, test_split=0.2):
    split_size = int(len(windows) * (1-test_split))
    train_windows = windows[:split_size]
    train_labels = labels[:split_size]
    test_windows = windows[split_size:]
    test_labels = labels[split_size:]
    return train_windows, test_windows, train_labels, test_labels


def evaluate_model(model, scaler, features, test_windows, test_labels, model_name, tracker, is_lstm=False):
    if is_lstm:
        # For LSTM, predict and inverse transform
        preds_scaled = model.predict(test_windows, verbose=0).reshape(-1)
        # Create full array for inverse transform
        preds_full = np.zeros((len(preds_scaled), len(features)))
        preds_full[:, features.index(TARGET_COL)] = preds_scaled
        for i, col in enumerate(features):
            if col != TARGET_COL:
                preds_full[:, i] = test_windows.iloc[:,
                                                     features.index(col)].values
        preds = scaler.inverse_transform(
            preds_full)[:, features.index(TARGET_COL)]
    else:
        # For sklearn models
        X_test_scaled = pd.DataFrame(
            scaler.transform(test_windows), columns=features)
        if TARGET_COL in X_test_scaled.columns:
            X_test_scaled = X_test_scaled.drop(columns=[TARGET_COL])
        preds = model.predict(X_test_scaled)

    timeseries_report_model(test_labels, preds, tracker,
                            model_name, seasonality=1, naive=False)


def main():
    print("Loading and preprocessing data...")
    df_h = load_and_preprocess_data()
    print(f"Data shape: {df_h.shape}")

    tracker = timeseries_models_tracker_df()

    # Naive model
    print("\n=== Modelo Ingenuo ===")
    y_true = df_h[TARGET_COL]
    x_true = df_h[f"{TARGET_COL} (-1h)"]
    split_size = int(len(y_true) * 0.9)
    y_test = y_true[split_size:]
    x_test = x_true[split_size:]
    timeseries_report_model(
        y_test, x_test, tracker, model_name="Modelo Ingenuo", seasonality=1, naive=True)

    # Load and evaluate each model
    models_to_evaluate = [
        ("XGBoost", "xgb_bundle.zip", load_xgb_bundle, False),
        ("LSTM Simple", "lstm_simple_bundle.zip", load_lstm_bundle, True),
        ("LSTM Full", "lstm_full_bundle.zip", load_lstm_bundle, True),
        ("LightGBM", "lgbm_bundle.zip", load_lgbm_bundle, False),
        ("PLS", "pls_bundle.zip", load_pls_bundle, False),
        ("LightGBM Full", "lgbm_full_bundle.zip", load_lgbm_bundle, False),
    ]

    for model_name, zip_path, load_func, is_lstm in models_to_evaluate:
        print(f"\n=== {model_name} ===")
        try:
            if is_lstm:
                model, features, scaler = load_func(
                    lstm_zip=zip_path, lstm_dir=zip_path.replace('.zip', ''))
            else:
                model, features, scaler = load_func(lgbm_zip=zip_path, lgbm_dir=zip_path.replace('.zip', '')) if 'lgbm' in zip_path else \
                    load_func(pls_zip=zip_path, pls_dir=zip_path.replace('.zip', '')) if 'pls' in zip_path else \
                    load_func(xgb_zip=zip_path,
                              xgb_dir=zip_path.replace('.zip', ''))

            # Prepare test data based on features
            if TARGET_COL in features:
                X = df_h[features].drop(TARGET_COL, axis=1)
                y = df_h[TARGET_COL]
            else:
                X = df_h[features]
                y = df_h[TARGET_COL]

            _, test_windows, _, test_labels = make_train_test_splits(
                X, y, test_split=0.1)

            evaluate_model(model, scaler, features, test_windows,
                           test_labels, model_name, tracker, is_lstm)
        except Exception as e:
            print(f"Error loading/evaluating {model_name}: {e}")

    print("\n=== Resumen de Rendimiento ===")
    print(tracker)


if __name__ == "__main__":
    main()
