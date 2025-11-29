# optimizer.py
# -*- coding: utf-8 -*-
"""
Optimizador +1h:
- Soporta múltiples motores: LSTM simple/full, XGB, LightGBM, PLS.
- Usa el engine especificado o auto para seleccionar el mejor disponible.
- Siempre reporta chequeo con XGB como validación si disponible.
"""

import warnings
import numpy as np
import pandas as pd
from utils import (
    preprocess_csv_to_hourly_features,
    load_xgb_bundle, prepare_xgb_matrix, to_dmatrix,
    load_lstm_bundle, prepare_lstm_inputs,
    load_lgbm_bundle, load_pls_bundle,
    get_reagent_cols, bounds_from_history,
    optimize_next_hour, backtest_optimizer, TARGET_COL
)

warnings.filterwarnings("ignore")

# ---- rutas locales ----
CSV_PATH = "MiningProcess_Flotation_Plant_Database.csv"
XGB_ZIP = "xgb_bundle.zip"
LSTM_ZIP = "lstm_bundle.zip"
LSTM_SIMPLE_ZIP = "lstm_simple_bundle.zip"
LSTM_FULL_ZIP = "lstm_full_bundle.zip"
LGBM_ZIP = "lgbm_bundle.zip"
LGBM_FULL_ZIP = "lgbm_full_bundle.zip"
PLS_ZIP = "pls_bundle.zip"


def main(engine="auto"):
    # 1) datos
    df_h = preprocess_csv_to_hourly_features(CSV_PATH)
    print("df_h listo:", df_h.shape)

    # 2) modelos
    xgb_model, xgb_features = load_xgb_bundle(
        xgb_zip=XGB_ZIP, xgb_dir="xgb_bundle")
    print(f"XGB features: {len(xgb_features)}")

    # LSTM bundles
    lstm_model = lstm_feats = lstm_scaler = None
    try:
        lstm_model, lstm_feats, lstm_scaler = load_lstm_bundle(
            lstm_zip=LSTM_ZIP, lstm_dir="lstm_bundle")
        print(f"LSTM features: {len(lstm_feats)}")
    except Exception as e:
        print("LSTM no disponible:", e)

    lstm_simple_model = lstm_simple_feats = lstm_simple_scaler = None
    try:
        lstm_simple_model, lstm_simple_feats, lstm_simple_scaler = load_lstm_bundle(
            lstm_zip=LSTM_SIMPLE_ZIP, lstm_dir="lstm_simple_bundle")
        print(f"LSTM simple features: {len(lstm_simple_feats)}")
    except Exception as e:
        print("LSTM simple no disponible:", e)

    lstm_full_model = lstm_full_feats = lstm_full_scaler = None
    try:
        lstm_full_model, lstm_full_feats, lstm_full_scaler = load_lstm_bundle(
            lstm_zip=LSTM_FULL_ZIP, lstm_dir="lstm_full_bundle")
        print(f"LSTM full features: {len(lstm_full_feats)}")
    except Exception as e:
        print("LSTM full no disponible:", e)

    # LightGBM bundles
    lgbm_model = lgbm_features = lgbm_scaler = None
    try:
        lgbm_model, lgbm_features, lgbm_scaler = load_lgbm_bundle(
            lgbm_zip=LGBM_ZIP, lgbm_dir="lgbm_bundle")
        print(f"LightGBM features: {len(lgbm_features)}")
    except Exception as e:
        print("LightGBM no disponible:", e)

    lgbm_full_model = lgbm_full_features = lgbm_full_scaler = None
    try:
        lgbm_full_model, lgbm_full_features, lgbm_full_scaler = load_lgbm_bundle(
            lgbm_zip=LGBM_FULL_ZIP, lgbm_dir="lgbm_full_bundle")
        print(f"LightGBM full features: {len(lgbm_full_features)}")
    except Exception as e:
        print("LightGBM full no disponible:", e)

    # PLS bundle
    pls_model = pls_features = pls_scaler = None
    try:
        pls_model, pls_features, pls_scaler = load_pls_bundle(
            pls_zip=PLS_ZIP, pls_dir="pls_bundle")
        print(f"PLS features: {len(pls_features)}")
    except Exception as e:
        print("PLS no disponible:", e)

    # 3) palancas y cotas (sin lags)
    REAGENT_COLS = get_reagent_cols(df_h.columns.tolist())
    if not REAGENT_COLS:
        raise RuntimeError(
            "No se detectaron palancas controlables (sin lags). Revisa patrones en get_reagent_cols().")
    BOUNDS = bounds_from_history(df_h, REAGENT_COLS, 2, 98)

    print("\nPalancas:")
    for c in REAGENT_COLS:
        lo, hi = BOUNDS[c]
        print(
            f" - {c}: [{lo:.4g} .. {hi:.4g}] (actual={df_h.iloc[-1].get(c, np.nan):.4g})")

    # Select LSTM model based on engine if specified
    selected_lstm = lstm_full_model
    selected_lstm_feats = lstm_full_feats
    selected_lstm_scaler = lstm_full_scaler
    if engine == "lstm_simple":
        selected_lstm = lstm_simple_model
        selected_lstm_feats = lstm_simple_feats
        selected_lstm_scaler = lstm_simple_scaler
    elif engine == "lstm_full":
        pass  # already set
    elif engine == "lstm" and lstm_model:  # legacy
        selected_lstm = lstm_model
        selected_lstm_feats = lstm_feats
        selected_lstm_scaler = lstm_scaler

    # 4) optimizar
    res = optimize_next_hour(
        df_h,
        reagent_cols=REAGENT_COLS,
        bounds_dict=BOUNDS,
        engine=engine,
        xgb_model=xgb_model, xgb_features=xgb_features,
        lstm_model=selected_lstm, lstm_feats=selected_lstm_feats, lstm_scaler=selected_lstm_scaler,
        lgbm_model=lgbm_full_model, lgbm_features=lgbm_full_features, lgbm_scaler=lgbm_full_scaler,
        pls_model=pls_model, pls_features=pls_features, pls_scaler=pls_scaler,
        maxiter=80, popsize=18, seed=42
    )

    print(f"\n=== Resultado optimización (+1h) | engine={res['engine']} ===")
    for c, v0, v1 in zip(res["reagent_cols"], res["baseline"], res["recommended"]):
        lo, hi = BOUNDS[c]
        print(f"{c:32s}: {v0:.4g} -> {v1:.4g}   [{lo:.4g}..{hi:.4g}]")
    if np.isfinite(res["pred_base_xgb"]) and np.isfinite(res["pred_opt_xgb"]):
        print(f"Pred base XGB : {res['pred_base_xgb']:.4f}")
        print(f"Pred ópt  XGB : {res['pred_opt_xgb']:.4f}")
        print(f"Mejora (XGB)  : {res['delta_xgb']:+.4f}")
    print(f"Aplicar en ventana [t, t+1] para t+1={res['t_next']}")

    # 5) (opcional) backtest corto con XGB (últimas 72 h)
    # Se demora mucho
    # try:
    #     bt = backtest_optimizer(
    #         df_h, xgb_model, xgb_features, REAGENT_COLS, BOUNDS,
    #         start=str(df_h.index[-72]), end=str(df_h.index[-1]),
    #         maxiter=30, popsize=12, seed=123
    #     )
    #     print("\nBacktest (últimas 72h):")
    #     print("Mejora media (opt - base):", (bt["opt_pred"] - bt["baseline_pred"]).mean())
    #     if bt["real_next"].notna().any():
    #         rmse_base = np.sqrt(np.nanmean((bt["baseline_pred"] - bt["real_next"])**2))
    #         rmse_opt  = np.sqrt(np.nanmean((bt["opt_pred"]    - bt["real_next"])**2))
    #         print("RMSE baseline vs real:", rmse_base)
    #         print("RMSE ópt vs real     :", rmse_opt)
    # except Exception as e:
    #     print("Backtest omitido:", e)


if __name__ == "__main__":
    # engine="auto" | "lstm_simple" | "lstm_full" | "xgb" | "lgbm" | "pls" | "lgbm_full"
    main(engine="auto")
