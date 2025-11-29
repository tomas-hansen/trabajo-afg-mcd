# utils.py
# -*- coding: utf-8 -*-
from scipy.optimize import differential_evolution
import os
import json
import zipfile
import warnings
import re
import numpy as np
import pandas as pd
import xgboost as xgb
from joblib import load as joblib_load

warnings.filterwarnings("ignore")
pd.options.display.max_columns = None

TARGET_COL = "% Iron Concentrate"

# -------------------------- patrones --------------------------
_LAG_PAT = re.compile(r"\(-\s*\d+\s*(?:mins?|min|h)\)", re.IGNORECASE)
_RE_PAT = re.compile(r"(starch|amine|amina|air\s*flow)", re.IGNORECASE)


def is_lagged_column(name: str) -> bool:
    if "(" in name and _LAG_PAT.search(name):
        return True
    return "(" in name  # fallback seguro

# -------------------------- Memoria --------------------------


def reduce_mem_usage(df, verbose=True):
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024.0**2
    for col in df.columns:
        t = df[col].dtypes
        if t in numerics:
            cmin, cmax = df[col].min(), df[col].max()
            if str(t)[:3] == "int":
                if cmin > np.iinfo(np.int8).min and cmax < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif cmin > np.iinfo(np.int16).min and cmax < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif cmin > np.iinfo(np.int32).min and cmax < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                else:
                    df[col] = df[col].astype(np.int64)
            else:
                if cmin > np.finfo(np.float16).min and cmax < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float32)
                elif cmin > np.finfo(np.float32).min and cmax < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
    end_mem = df.memory_usage().sum() / 1024.0**2
    if verbose and start_mem > 0:
        print(
            f"Memoria: {start_mem:.2f}MB -> {end_mem:.2f}MB ({100*(start_mem-end_mem)/start_mem:.1f}% menos)")
    return df

# -------------------------- Helpers internos --------------------------


def _inject_duplicate_row_gap_fix(df):
    df_before = df.copy().loc[: "2017-04-10 00:00:00"]
    df_after = df.copy().loc["2017-04-10 01:00:00":]
    new_date = pd.to_datetime("2017-04-10 00:00:00")
    new_data = pd.DataFrame(
        df_before[-1:].values, index=[new_date], columns=df_before.columns)
    return pd.concat([pd.concat([df_before, new_data]), df_after])


def _build_monotonic_seconds_index(df):
    df = df.reset_index(allow_duplicates=True)
    df["duration"] = 20
    df.loc[0, "duration"] = 0
    df["duration"] = df["duration"].cumsum()
    df["Date_with_seconds"] = pd.Timestamp(
        "2017-03-29 12:00:00") + pd.to_timedelta(df["duration"], unit="s")
    df = df.set_index("Date_with_seconds").drop(columns=["index", "duration"])
    return df


def _add_time_features_hourly(df_h):
    df_h = df_h.copy()
    df_h.index.names = ["Date"]
    df_h["hora"] = df_h.index.hour
    df_h["sin_hora"] = np.sin(2 * np.pi * df_h["hora"] / 24)
    df_h["cos_hora"] = np.cos(2 * np.pi * df_h["hora"] / 24)
    df_h["dia_de_la_semana"] = df_h.index.day_of_week
    df_h["sin_dia_de_la_semana"] = np.sin(
        2 * np.pi * df_h["dia_de_la_semana"] / 7)
    df_h["cos_dia_de_la_semana"] = np.cos(
        2 * np.pi * df_h["dia_de_la_semana"] / 7)
    feat_eng_vars = ["sin_hora", "cos_hora",
                     "sin_dia_de_la_semana", "cos_dia_de_la_semana"]
    feat_eng_df = df_h[feat_eng_vars].copy()
    df_h = df_h.drop(["hora", "dia_de_la_semana"], axis=1)
    return df_h, feat_eng_df

# -------------------------- Preprocesamiento principal --------------------------


def preprocess_csv_to_hourly_features(csv_path, decimal=",", start_ts="2017-03-29 12:00:00", target_col=TARGET_COL):
    df = pd.read_csv(csv_path, decimal=decimal,
                     parse_dates=["date"], index_col="date")
    df = df.loc[start_ts:]
    df = reduce_mem_usage(df, verbose=False)

    df = _inject_duplicate_row_gap_fix(df)
    df = _build_monotonic_seconds_index(df)

    df_h = df.resample("H").first()
    df_h, feat_eng_df = _add_time_features_hourly(df_h)

    list_cols = [c for c in df.columns.to_list()]
    for tcol in ["% Silica Concentrate", target_col]:
        if tcol in list_cols:
            list_cols.remove(tcol)

    df_15 = df.resample("15min").first().drop(
        "% Silica Concentrate", axis=1, errors="ignore")
    for i_col in list_cols:
        for i in range(3):
            df_15[f"{i_col} ({-15 * (i + 1)}mins)"] = df_15[i_col].shift(i + 1)

    df_h2 = df_15.resample("H").first()
    df_h2.index.names = ["Date"]

    for i in range(3):
        df_h2[f"{target_col} ({-i-1}h)"] = df_h2[target_col].shift(i + 1)

    df_h = df_h2.join(feat_eng_df, how="left")
    cols = df_h.columns.to_list()
    if target_col in cols:
        cols.remove(target_col)
        cols.insert(0, target_col)
        df_h = df_h[cols]
    df_h = df_h.dropna().astype("float32")
    return df_h

# -------------------------- ZIP helper --------------------------


def ensure_unzipped(zip_path, target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir, exist_ok=True)
    has_content = any(os.scandir(target_dir))
    if (not has_content) and os.path.isfile(zip_path):
        with zipfile.ZipFile(zip_path, "r") as zf:
            zf.extractall(target_dir)

# -------------------------- XGBoost bundle --------------------------


def load_xgb_bundle(xgb_dir="xgb_bundle", xgb_zip=None, model_filename="model.json", features_filename="feature_names.json"):
    if xgb_zip:
        ensure_unzipped(xgb_zip, xgb_dir)
    xgb_model_path = os.path.join(xgb_dir, model_filename)
    xgb_feats_path = os.path.join(xgb_dir, features_filename)
    model = xgb.Booster()
    model.load_model(xgb_model_path)
    with open(xgb_feats_path, "r", encoding="utf-8") as f:
        feature_names = json.load(f)
    return model, feature_names


def prepare_xgb_matrix(df_h, xgb_features, target_col=TARGET_COL):
    X = df_h.copy()
    if target_col in X.columns:
        X = X.drop(columns=[target_col])
    for c in xgb_features:
        if c not in X.columns:
            X[c] = 0.0
    X = X[list(xgb_features)].astype("float32")
    return X


def to_dmatrix(X):
    return xgb.DMatrix(X)

# -------------------------- LSTM bundle --------------------------


def load_lstm_bundle(lstm_dir="lstm_bundle", lstm_zip=None, model_filename="model.keras",
                     features_filename="feature_names.json", scaler_filename="scaler.joblib"):
    if lstm_zip:
        ensure_unzipped(lstm_zip, lstm_dir)
    lstm_model_path = os.path.join(lstm_dir, model_filename)
    lstm_feats_path = os.path.join(lstm_dir, features_filename)
    lstm_scaler_path = os.path.join(lstm_dir, scaler_filename)
    from tensorflow import keras
    model = keras.models.load_model(lstm_model_path, safe_mode=False)
    with open(lstm_feats_path, "r", encoding="utf-8") as f:
        lstm_feats = json.load(f)
    scaler = joblib_load(lstm_scaler_path)
    return model, lstm_feats, scaler


def prepare_lstm_inputs(df_h, lstm_feats, scaler, target_col=TARGET_COL):
    use_cols = [c for c in lstm_feats if c in df_h.columns]
    df_lstm_only = df_h[use_cols].copy()
    df_lstm_scaled = pd.DataFrame(scaler.transform(
        df_lstm_only), index=df_lstm_only.index, columns=lstm_feats)
    if target_col in df_lstm_scaled.columns:
        X_lstm_infer = df_lstm_scaled.drop(
            columns=[target_col]).astype("float32")
    else:
        X_lstm_infer = df_lstm_scaled.astype("float32")
    return df_lstm_only, df_lstm_scaled, X_lstm_infer
# -------------------------- LightGBM bundle --------------------------


def load_lgbm_bundle(lgbm_dir="lgbm_bundle", lgbm_zip=None, model_filename="model.joblib",
                     features_filename="feature_names.json", scaler_filename="scaler.joblib"):
    if lgbm_zip:
        ensure_unzipped(lgbm_zip, lgbm_dir)
    lgbm_model_path = os.path.join(lgbm_dir, model_filename)
    lgbm_feats_path = os.path.join(lgbm_dir, features_filename)
    lgbm_scaler_path = os.path.join(lgbm_dir, scaler_filename)
    model = joblib_load(lgbm_model_path)
    with open(lgbm_feats_path, "r", encoding="utf-8") as f:
        lgbm_feats = json.load(f)
    scaler = joblib_load(lgbm_scaler_path)
    return model, lgbm_feats, scaler

# -------------------------- PLS bundle --------------------------


def load_pls_bundle(pls_dir="pls_bundle", pls_zip=None, model_filename="model.joblib",
                    features_filename="feature_names.json", scaler_filename="scaler.joblib"):
    if pls_zip:
        ensure_unzipped(pls_zip, pls_dir)
    pls_model_path = os.path.join(pls_dir, model_filename)
    pls_feats_path = os.path.join(pls_dir, features_filename)
    pls_scaler_path = os.path.join(pls_dir, scaler_filename)
    model = joblib_load(pls_model_path)
    with open(pls_feats_path, "r", encoding="utf-8") as f:
        pls_feats = json.load(f)
    scaler = joblib_load(pls_scaler_path)
    return model, pls_feats, scaler


def prepare_sklearn_inputs(df_h, features, scaler, target_col=TARGET_COL):
    use_cols = [c for c in features if c in df_h.columns]
    df_only = df_h[use_cols].copy()
    df_scaled = pd.DataFrame(scaler.transform(
        df_only), index=df_only.index, columns=features)
    if target_col in df_scaled.columns:
        X_infer = df_scaled.drop(columns=[target_col]).astype("float32")
    else:
        X_infer = df_scaled.astype("float32")
    return df_only, df_scaled, X_infer

# -------------------------- Palancas y cotas --------------------------


def get_reagent_cols(df_cols):
    """Solo columnas controlables 'instantáneas' (sin lags)."""
    cols = []
    for c in df_cols:
        if c == TARGET_COL:
            continue
        if _RE_PAT.search(c) and not is_lagged_column(c):
            cols.append(c)
    seen, out = set(), []
    for c in cols:
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def bounds_from_history(df_h, cols, p_low=5, p_high=95):
    b = {}
    for c in cols:
        s = pd.to_numeric(df_h[c], errors="coerce").dropna()
        if len(s) == 0:
            b[c] = (-1.0, 1.0)
            continue
        lo, hi = np.percentile(s.values, [p_low, p_high])
        if lo == hi:
            lo, hi = float(s.min()), float(s.max())
        b[c] = (float(lo), float(hi))
    return b

# -------------------------- Construcción de features (XGB) --------------------------


def make_objective_sklearn(df_h, model, features, scaler, reagent_cols, bounds_dict, target_col=TARGET_COL):
    last_ts = df_h.index[-1]
    last_row = df_h.iloc[-1]

    def predict_with_sklearn(x):
        xr = []
        for i, c in enumerate(reagent_cols):
            lo, hi = bounds_dict[c]
            xr.append(float(np.clip(x[i], lo, hi)))
        X_next = build_next_hour_features(last_row, last_ts, np.array(
            xr, dtype=np.float32), reagent_cols, features)
        # Scale
        X_scaled = pd.DataFrame(scaler.transform(X_next), columns=features)
        if target_col in X_scaled.columns:
            X_scaled = X_scaled.drop(columns=[target_col])
        return float(model.predict(X_scaled)[0])

    def objective(x):
        return -predict_with_sklearn(x)
    return objective


def build_next_hour_features(last_row, last_ts, candidate_vec, reagent_cols, xgb_features, target_col=TARGET_COL):
    row = last_row.copy()
    t_next = last_ts + pd.Timedelta(hours=1)
    h = t_next.hour
    d = t_next.dayofweek
    if "sin_hora" in row.index:
        row["sin_hora"] = np.sin(2*np.pi*h/24.0)
    if "cos_hora" in row.index:
        row["cos_hora"] = np.cos(2*np.pi*h/24.0)
    if "sin_dia_de_la_semana" in row.index:
        row["sin_dia_de_la_semana"] = np.sin(2*np.pi*d/7.0)
    if "cos_dia_de_la_semana" in row.index:
        row["cos_dia_de_la_semana"] = np.cos(2*np.pi*d/7.0)
    for c, v in zip(reagent_cols, candidate_vec):
        if (c in row.index) and (not is_lagged_column(c)):
            row[c] = float(v)
    X_row = pd.DataFrame([row], index=[t_next])
    if target_col in X_row.columns:
        X_row = X_row.drop(columns=[target_col])
    for c in xgb_features:
        if c not in X_row.columns:
            X_row[c] = 0.0
    X_row = X_row[xgb_features].astype("float32")
    return X_row

# -------------------------- Objetivos de optimización --------------------------


def make_objective_xgb(df_h, xgb_model, xgb_features, reagent_cols, bounds_dict):
    last_ts = df_h.index[-1]
    last_row = df_h.iloc[-1]

    def predict_with_xgb(x):
        xr = []
        for i, c in enumerate(reagent_cols):
            lo, hi = bounds_dict[c]
            xr.append(float(np.clip(x[i], lo, hi)))
        X_next = build_next_hour_features(last_row, last_ts, np.array(
            xr, dtype=np.float32), reagent_cols, xgb_features)
        return float(xgb_model.predict(xgb.DMatrix(X_next))[0])

    def objective(x):
        return -predict_with_xgb(x)
    return objective


def make_objective_lstm(df_h, lstm_model, lstm_feats, lstm_scaler, reagent_cols, bounds_dict, target_col=TARGET_COL):
    """
    Usa LSTM como función objetivo sólo si las palancas existen entre lstm_feats.
    Modifica la última fila escalada en las columnas de palancas y predice t+1.
    """
    # verificar que al menos una palanca esté en features LSTM
    palancas_en_lstm = [c for c in reagent_cols if c in lstm_feats]
    last_idx = df_h.index[-1]

    # preparar última fila (escalada)
    use_cols = [c for c in lstm_feats if c in df_h.columns]
    base_row = df_h.loc[last_idx, use_cols].copy()
    base_scaled = pd.Series(lstm_scaler.transform(
        base_row.to_frame().T)[0], index=lstm_feats)

    def predict_with_lstm(x):
        row_scaled = base_scaled.copy()
        # inyectar palancas (si existen en LSTM)
        for i, c in enumerate(reagent_cols):
            if c in row_scaled.index:
                lo, hi = bounds_dict[c]
                v = float(np.clip(x[i], lo, hi))
                # proyectar valor crudo a escala del scaler: necesitamos escalar en el espacio original
                # -> desescalar base a original, reemplazar, y reescalar:
                base_orig = base_row.copy()
                base_orig[c] = v
                row_scaled_local = pd.Series(lstm_scaler.transform(
                    base_orig.to_frame().T)[0], index=lstm_feats)
                row_scaled[c] = row_scaled_local[c]
        # construir X (drop target si está)
        if target_col in row_scaled.index:
            X_vec = row_scaled.drop(labels=[target_col]).values.astype(
                "float32").reshape(1, -1)
        else:
            X_vec = row_scaled.values.astype("float32").reshape(1, -1)
        pred_scaled = float(lstm_model.predict(
            X_vec, verbose=0).reshape(-1)[0])

        # desescalar predicción a unidades reales reconstruyendo vector completo
        recon = row_scaled.copy()
        recon[target_col] = pred_scaled
        recon_df = pd.DataFrame([recon.values], columns=lstm_feats, index=[
                                last_idx + pd.Timedelta(hours=1)])
        pred_real = lstm_scaler.inverse_transform(
            recon_df.values)[0, lstm_feats.index(target_col)]
        return pred_real

    def objective(x):
        return -predict_with_lstm(x)

    return objective, (len(palancas_en_lstm) > 0)


# -------------------------- Optimización (DE) --------------------------


def optimize_next_hour(df_h, reagent_cols, bounds_dict,
                       engine="auto",
                       xgb_model=None, xgb_features=None,
                       lstm_model=None, lstm_feats=None, lstm_scaler=None,
                       lgbm_model=None, lgbm_features=None, lgbm_scaler=None,
                       pls_model=None, pls_features=None, pls_scaler=None,
                       maxiter=80, popsize=18, seed=42):
    """
    engine: 'auto' | 'lstm_simple' | 'lstm_full' | 'xgb' | 'lgbm' | 'pls' | 'lgbm_full'
    """
    bounds_list = [bounds_dict[c] for c in reagent_cols]

    # Determine which model to use
    if engine == "auto":
        # Prioritize LSTM full, then others
        if lstm_model and lstm_feats and lstm_scaler:
            engine = "lstm_full"
        elif xgb_model and xgb_features:
            engine = "xgb"
        elif lgbm_model and lgbm_features and lgbm_scaler:
            engine = "lgbm"
        elif pls_model and pls_features and pls_scaler:
            engine = "pls"
        else:
            raise ValueError("No model available for auto selection.")

    if engine in ["lstm_simple", "lstm_full"]:
        if not (lstm_model and lstm_feats and lstm_scaler):
            raise ValueError(f"LSTM model not available for engine {engine}.")
        objective, ok = make_objective_lstm(
            df_h, lstm_model, lstm_feats, lstm_scaler, reagent_cols, bounds_dict)
        if not ok:
            raise ValueError(
                f"LSTM {engine} does not have controllable features.")
    elif engine == "xgb":
        if not (xgb_model and xgb_features):
            raise ValueError("XGBoost model not available.")
        objective = make_objective_xgb(
            df_h, xgb_model, xgb_features, reagent_cols, bounds_dict)
    elif engine in ["lgbm", "lgbm_full"]:
        if not (lgbm_model and lgbm_features and lgbm_scaler):
            raise ValueError("LightGBM model not available.")
        objective = make_objective_sklearn(
            df_h, lgbm_model, lgbm_features, lgbm_scaler, reagent_cols, bounds_dict)
    elif engine == "pls":
        if not (pls_model and pls_features and pls_scaler):
            raise ValueError("PLS model not available.")
        objective = make_objective_sklearn(
            df_h, pls_model, pls_features, pls_scaler, reagent_cols, bounds_dict)
    else:
        raise ValueError(f"Unknown engine: {engine}")

    res = differential_evolution(objective, bounds_list, strategy="best1bin",
                                 maxiter=maxiter, popsize=popsize, tol=1e-6,
                                 mutation=(0.5, 1.0), recombination=0.7,
                                 polish=True, seed=seed)

    bestx = []
    for v, c in zip(res.x, reagent_cols):
        lo, hi = bounds_dict[c]
        bestx.append(float(np.clip(v, lo, hi)))
    bestx = np.array(bestx, dtype=np.float32)

    # baseline (valores actuales)
    last_row = df_h.iloc[-1]
    base_vec = np.array([last_row.get(c, np.nan)
                        for c in reagent_cols], dtype=np.float32)

    # evaluar con XGB (para reporte, si disponible)
    pred_base = pred_opt = np.nan
    if (xgb_model is not None) and (xgb_features is not None):
        last_ts = df_h.index[-1]
        X_base = build_next_hour_features(
            last_row, last_ts, base_vec, reagent_cols, xgb_features)
        X_opt = build_next_hour_features(
            last_row, last_ts, bestx,   reagent_cols, xgb_features)
        pred_base = float(xgb_model.predict(xgb.DMatrix(X_base))[0])
        pred_opt = float(xgb_model.predict(xgb.DMatrix(X_opt))[0])

    return {
        "engine": engine,
        "t_next": df_h.index[-1] + pd.Timedelta(hours=1),
        "reagent_cols": reagent_cols,
        "recommended": bestx,
        "baseline": base_vec,
        "pred_base_xgb": pred_base,
        "pred_opt_xgb": pred_opt,
        "delta_xgb": (pred_opt - pred_base) if (np.isfinite(pred_base) and np.isfinite(pred_opt)) else np.nan
    }

# -------------------------- Backtest sencillo (XGB) --------------------------


def backtest_optimizer(df_h, xgb_model, xgb_features, reagent_cols, bounds_dict,
                       start=None, end=None, maxiter=40, popsize=14, seed=123):
    idx = df_h.loc[start:end].index if (start or end) else df_h.index
    rows = []
    for t0 in idx[:-1]:
        r0 = df_h.loc[t0]
        base_vec = np.array([r0.get(c, np.nan)
                            for c in reagent_cols], dtype=np.float32)
        if np.isnan(base_vec).any():
            continue

        def obj_local(x):
            xr = []
            for i, c in enumerate(reagent_cols):
                lo, hi = bounds_dict[c]
                xr.append(float(np.clip(x[i], lo, hi)))
            Xn = build_next_hour_features(r0, t0, np.array(
                xr, dtype=np.float32), reagent_cols, xgb_features)
            return -float(xgb_model.predict(xgb.DMatrix(Xn))[0])
        bounds_list = [bounds_dict[c] for c in reagent_cols]
        res = differential_evolution(obj_local, bounds_list, maxiter=maxiter, popsize=popsize,
                                     tol=1e-6, mutation=(0.5, 1.0), recombination=0.7,
                                     polish=False, seed=seed)
        bestx = []
        for v, c in zip(res.x, reagent_cols):
            lo, hi = bounds_dict[c]
            bestx.append(float(np.clip(v, lo, hi)))
        bestx = np.array(bestx, dtype=np.float32)

        Xb = build_next_hour_features(
            r0, t0, base_vec, reagent_cols, xgb_features)
        Xo = build_next_hour_features(
            r0, t0, bestx,   reagent_cols, xgb_features)

        base_pred = float(xgb_model.predict(xgb.DMatrix(Xb))[0])
        opt_pred = float(xgb_model.predict(xgb.DMatrix(Xo))[0])

        t1 = t0 + pd.Timedelta(hours=1)
        real_next = np.nan
        if (TARGET_COL in df_h.columns) and (t1 in df_h.index):
            real_next = float(df_h.loc[t1][TARGET_COL])
        rows.append({"t0": t0, "t1": t1, "baseline_pred": base_pred,
                    "opt_pred": opt_pred, "real_next": real_next})
    return pd.DataFrame(rows)
