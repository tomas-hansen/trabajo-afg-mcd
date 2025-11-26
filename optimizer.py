# optimizer.py
# -*- coding: utf-8 -*-
"""
Optimizador +1h:
- Usa LSTM como motor si el bundle LSTM incluye palancas en sus features; si no, cae a XGB.
- Siempre reporta chequeo instantáneo con XGB como validación.
"""

import warnings, numpy as np, pandas as pd
from utils import (
    preprocess_csv_to_hourly_features,
    load_xgb_bundle, prepare_xgb_matrix, to_dmatrix,
    load_lstm_bundle, prepare_lstm_inputs,
    get_reagent_cols, bounds_from_history,
    optimize_next_hour, backtest_optimizer, TARGET_COL
)

warnings.filterwarnings("ignore")

# ---- rutas locales ----
CSV_PATH = "MiningProcess_Flotation_Plant_Database.csv"
XGB_ZIP  = "xgb_bundle.zip"
LSTM_ZIP = "lstm_bundle.zip"

def main(engine="auto"):
    # 1) datos
    df_h = preprocess_csv_to_hourly_features(CSV_PATH)
    print("df_h listo:", df_h.shape)

    # 2) modelos
    xgb_model, xgb_features = load_xgb_bundle(xgb_zip=XGB_ZIP, xgb_dir="xgb_bundle")
    print(f"XGB features: {len(xgb_features)}")
    try:
        lstm_model, lstm_feats, lstm_scaler = load_lstm_bundle(lstm_zip=LSTM_ZIP, lstm_dir="lstm_bundle")
        print(f"LSTM features: {len(lstm_feats)}")
    except Exception as e:
        lstm_model = lstm_feats = lstm_scaler = None
        print("LSTM no disponible o no cargó:", e)

    # 3) palancas y cotas (sin lags)
    REAGENT_COLS = get_reagent_cols(df_h.columns.tolist())
    if not REAGENT_COLS:
        raise RuntimeError("No se detectaron palancas controlables (sin lags). Revisa patrones en get_reagent_cols().")
    BOUNDS = bounds_from_history(df_h, REAGENT_COLS, 2, 98)

    print("\nPalancas:")
    for c in REAGENT_COLS:
        lo, hi = BOUNDS[c]
        print(f" - {c}: [{lo:.4g} .. {hi:.4g}] (actual={df_h.iloc[-1].get(c, np.nan):.4g})")

    # 4) optimizar
    res = optimize_next_hour(
        df_h,
        reagent_cols=REAGENT_COLS,
        bounds_dict=BOUNDS,
        engine=engine,
        xgb_model=xgb_model, xgb_features=xgb_features,
        lstm_model=lstm_model, lstm_feats=lstm_feats, lstm_scaler=lstm_scaler,
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
    #Se demora mucho
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
    # engine="auto" | "lstm" | "xgb"
    main(engine="lstm")
