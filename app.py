# app.py — Advanced Dashboard (ARIMA + ML + Hybrid)
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import json
import os
import io
import matplotlib.pyplot as plt
from pandas.tseries.offsets import MonthBegin
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.graphics.tsaplots import plot_acf

# -----------------------------
# Page setup / theme
# -----------------------------
st.set_page_config(
    page_title="Yes Bank Forecast – ARIMA • ML • Hybrid",
    page_icon="📈",
    layout="wide"
)
st.markdown(
    """
    <style>
    .kpi {background:#0e1117;border:1px solid #2b2f36;border-radius:12px;padding:14px 16px;}
    .kpi h3 {margin:0;font-size:16px;color:#9aa4b2;font-weight:600;}
    .kpi p {margin:4px 0 0 0;font-size:22px;font-weight:700;color:white;}
    .warn {color:#ffcc00;}
    </style>
    """,
    unsafe_allow_html=True
)

st.title("📈 Yes Bank Stock Forecasting — ARIMA • ML • Hybrid")

# -----------------------------
# Paths (expect these files next to app.py)
# -----------------------------
DATA_PATH            = "data.csv"
ARIMA_PATH           = "arima_model.pkl"
ML_PATH              = "best_ml_model.pkl"          # trend model (e.g., RandomForest / XGB)
RESID_MODEL_PATH     = "rf_resid.pkl"               # residual model
SCALER_PATH          = "scaler.pkl"
SCALER_RESID_PATH    = "scaler_resid.pkl"
FEATURE_JSON_1       = "features.json"
FEATURE_JSON_2       = "feature_cols.json"          # accept either name

# -----------------------------
# Helpers
# -----------------------------
def load_json_any(paths):
    for p in paths:
        if os.path.exists(p):
            with open(p, "r") as f:
                return json.load(f)
    return None

@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["Date"])
    # keep OHLC if present; ensure Close numeric
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Close"])
    df = df.sort_values("Date")
    df.set_index("Date", inplace=True)
    # normalize to month start
    df.index = df.index.to_period("M").to_timestamp()
    return df

def ensure_features(df: pd.DataFrame, feature_cols: list) -> pd.DataFrame:
    """Create any missing engineered feature columns required by ML models."""
    # Always compute these if missing
    if "Month" in feature_cols and "Month" not in df.columns:
        df["Month"] = df.index.month
    if "Year" in feature_cols and "Year" not in df.columns:
        df["Year"] = df.index.year
    if "Range" in feature_cols and "Range" not in df.columns:
        if ("High" in df.columns) and ("Low" in df.columns):
            df["Range"] = df["High"] - df["Low"]
        else:
            df["Range"] = df["Close"].rolling(3).std()
    if "Lag1" in feature_cols and "Lag1" not in df.columns:
        df["Lag1"] = df["Close"].shift(1)
    if "Lag2" in feature_cols and "Lag2" not in df.columns:
        df["Lag2"] = df["Close"].shift(2)
    if "RollMean3" in feature_cols and "RollMean3" not in df.columns:
        df["RollMean3"] = df["Close"].rolling(3).mean()
    # Fraud Event indicator (structural break in 2018)
    if "FraudEvent" in feature_cols and "FraudEvent" not in df.columns:
        df["FraudEvent"] = (df.index >= '2018-01-01').astype(int)

    # After creating, drop initial NaNs from lags/rolling to keep model input clean
    df = df.dropna()
    return df

def make_future_dates(last_ts, steps: int):
    start = pd.to_datetime(last_ts) + MonthBegin()
    return pd.date_range(start=start, periods=steps, freq="MS")

def kpi_card(title: str, value: float, suffix=""):
    col = st.container()
    with col:
        st.markdown(f'<div class="kpi"><h3>{title}</h3><p>{value:.3f}{suffix}</p></div>', unsafe_allow_html=True)

def calc_metrics(y_true, y_pred):
    mae = np.mean(np.abs(y_true - y_pred))
    rmse = np.sqrt(np.mean((y_true - y_pred)**2))
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, rmse, mape

def arima_order_from(model) -> tuple:
    try:
        mo = model.model_orders
        return (mo.get("ar", 0), model.model.k_diff, mo.get("ma", 0))
    except Exception:
        # Fall back if not available
        return (1, 1, 1)

# -----------------------------
# Load all assets
# -----------------------------
if not os.path.exists(DATA_PATH):
    st.error("`data.csv` is missing. Put it next to `app.py` and rerun.")
    st.stop()
df_full = load_data(DATA_PATH)

# feature list
feature_cols = load_json_any([FEATURE_JSON_1, FEATURE_JSON_2])
if feature_cols is None:
    # Fall back to the standard 6 if json missing
    feature_cols = ['Month', 'Year', 'Range', 'Lag1', 'Lag2', 'RollMean3']

# models
if not os.path.exists(ARIMA_PATH):
    st.error("`arima_model.pkl` is missing.")
    st.stop()
arima_model = joblib.load(ARIMA_PATH)

ml_model = None
scaler = None
resid_model = None
scaler_resid = None

if os.path.exists(ML_PATH):
    ml_model = joblib.load(ML_PATH)
if os.path.exists(SCALER_PATH):
    scaler = joblib.load(SCALER_PATH)

if os.path.exists(RESID_MODEL_PATH):
    resid_model = joblib.load(RESID_MODEL_PATH)
if os.path.exists(SCALER_RESID_PATH):
    scaler_resid = joblib.load(SCALER_RESID_PATH)

# Build engineered frame for ML usage
df_ml = ensure_features(df_full.copy(), feature_cols)

# -----------------------------
# Sidebar controls
# -----------------------------
st.sidebar.header("Controls")
horizon = st.sidebar.selectbox("Forecast horizon (months)", [12, 24, 36], index=0)
model_choice = st.sidebar.selectbox(
    "Model to display",
    [m for m in ["Hybrid", "ARIMA", "ML"] if not (m == "ML" and (ml_model is None or scaler is None))]
)

# -----------------------------
# Forecast functions (ARIMA / ML / Hybrid)
# -----------------------------
def forecast_arima(steps: int):
    dates = make_future_dates(df_full.index[-1], steps)
    preds = arima_model.forecast(steps=steps)
    preds.index = dates
    return preds

def forecast_ml(steps: int):
    if ml_model is None or scaler is None:
        return None
    dates = make_future_dates(df_ml.index[-1], steps)
    preds = []
    last_close = df_ml["Close"].iloc[-2:].tolist()
    # Use last observed Range as constant for the horizon (pragmatic choice)
    last_range = float(df_ml["Range"].iloc[-1]) if "Range" in df_ml.columns else float(df_ml['Close'].rolling(3).std().iloc[-1])

    base = df_ml[feature_cols].iloc[-1:].copy()
    for dt in dates:
        lag1 = last_close[-1]
        lag2 = last_close[-2]
        rollmean3 = np.mean([lag1, lag2, lag1])

        row = base.copy()
        if "Lag1" in row.columns:      row.iloc[0, row.columns.get_loc("Lag1")] = lag1
        if "Lag2" in row.columns:      row.iloc[0, row.columns.get_loc("Lag2")] = lag2
        if "RollMean3" in row.columns: row.iloc[0, row.columns.get_loc("RollMean3")] = rollmean3
        if "Month" in row.columns:     row.iloc[0, row.columns.get_loc("Month")] = dt.month
        if "Year" in row.columns:      row.iloc[0, row.columns.get_loc("Year")]  = dt.year
        if "Range" in row.columns:     row.iloc[0, row.columns.get_loc("Range")] = last_range
        if "FraudEvent" in row.columns:
            row.iloc[0, row.columns.get_loc("FraudEvent")] = 1 if dt >= pd.Timestamp("2018-01-01") else 0

        Xs = scaler.transform(row[feature_cols])
        yhat = ml_model.predict(Xs)[0]
        preds.append(yhat)

        last_close.append(yhat); last_close.pop(0)

    return pd.Series(preds, index=dates)

def forecast_hybrid(steps: int):
    if resid_model is None or scaler_resid is None:
        return None
    # ARIMA mean
    arima_mean = forecast_arima(steps)
    dates = arima_mean.index

    # residuals predicted on hybrid lags
    preds_resid = []
    last_hybrid_closes = df_ml["Close"].iloc[-2:].tolist()
    last_range = float(df_ml["Range"].iloc[-1]) if "Range" in df_ml.columns else float(df_ml['Close'].rolling(3).std().iloc[-1])
    base = df_ml[feature_cols].iloc[-1:].copy()

    for dt, a_mean in zip(dates, arima_mean.values):
        lag1 = last_hybrid_closes[-1]
        lag2 = last_hybrid_closes[-2]
        rollmean3 = np.mean([lag1, lag2, lag1])

        row = base.copy()
        if "Lag1" in row.columns:      row.iloc[0, row.columns.get_loc("Lag1")] = lag1
        if "Lag2" in row.columns:      row.iloc[0, row.columns.get_loc("Lag2")] = lag2
        if "RollMean3" in row.columns: row.iloc[0, row.columns.get_loc("RollMean3")] = rollmean3
        if "Month" in row.columns:     row.iloc[0, row.columns.get_loc("Month")] = dt.month
        if "Year" in row.columns:      row.iloc[0, row.columns.get_loc("Year")]  = dt.year
        if "Range" in row.columns:     row.iloc[0, row.columns.get_loc("Range")] = last_range
        if "FraudEvent" in row.columns:
            row.iloc[0, row.columns.get_loc("FraudEvent")] = 1 if dt >= pd.Timestamp("2018-01-01") else 0

        Xs = scaler_resid.transform(row[feature_cols])
        rhat = resid_model.predict(Xs)[0]
        preds_resid.append(rhat)

        # update hybrid lag with (ARIMA mean + residual)
        hybrid_now = a_mean + rhat
        last_hybrid_closes.append(hybrid_now); last_hybrid_closes.pop(0)

    hybrid = arima_mean.values + np.array(preds_resid)
    return pd.Series(hybrid, index=dates), arima_mean, pd.Series(preds_resid, index=dates)

# -----------------------------
# Build forecasts
# -----------------------------
arima_fc = forecast_arima(horizon)
ml_fc    = forecast_ml(horizon) if (ml_model and scaler) else None
hyb_fc   = None
hyb_parts = None
if resid_model and scaler_resid:
    hyb_fc, hyb_arima_part, hyb_resid_part = forecast_hybrid(horizon)
    hyb_parts = (hyb_arima_part, hyb_resid_part)

# -----------------------------
# Tabs
# -----------------------------
tab_overview, tab_forecast, tab_bt, tab_resid, tab_feat, tab_data, tab_dl = st.tabs(
    ["Overview", "Forecast", "Backtest", "Residuals", "Feature Importance", "Data Explorer", "Download Center"]
)

# -----------------------------
# OVERVIEW
# -----------------------------
with tab_overview:
    st.subheader("At a glance")
    c1, c2, c3 = st.columns(3)
    # naive “volatility” proxy = last 12m std
    vol = float(df_full["Close"].tail(12).std()) if len(df_full) >= 12 else float(df_full["Close"].std())
    kpi_card("Last Close", df_full["Close"].iloc[-1])
    with c2:
        st.markdown(f'<div class="kpi"><h3>12-month σ (volatility)</h3><p>{vol:.3f}</p></div>', unsafe_allow_html=True)
    with c3:
        kpi_card("History points", len(df_full), "")

    st.markdown("—")
    st.markdown("**Historical Close**")
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(df_full.index, df_full["Close"], color="black")
    ax.set_title("Monthly Close")
    ax.grid(alpha=.3)
    st.pyplot(fig)

# -----------------------------
# FORECAST
# -----------------------------
with tab_forecast:
    st.subheader("Forecast comparison")
    st.caption(f"Horizon: **{horizon} months** · Model shown: **{model_choice}**")

    # assemble table
    out = pd.DataFrame(index=arima_fc.index)
    out["ARIMA"] = arima_fc.values
    if ml_fc is not None:
        out["ML"] = ml_fc.values
    if hyb_fc is not None:
        out["Hybrid"] = hyb_fc.values

    st.dataframe(out.round(4))

    # plot
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(df_full.index, df_full["Close"], label="Historical", color="black", linewidth=2)
    if model_choice in ["ARIMA", "Hybrid"]:
        ax.plot(arima_fc.index, arima_fc.values, "--o", label="ARIMA")
    if ml_fc is not None and model_choice in ["ML", "Hybrid"]:
        ax.plot(ml_fc.index, ml_fc.values, "--x", label="ML")
    if hyb_fc is not None and model_choice in ["Hybrid"]:
        ax.plot(hyb_fc.index, hyb_fc.values, "--s", label="Hybrid")
    ax.set_title("Forecast")
    ax.set_xlabel("Date"); ax.set_ylabel("Close")
    ax.grid(alpha=.3); ax.legend()
    st.pyplot(fig)

    if model_choice == "Hybrid" and hyb_parts is not None:
        st.markdown("**Hybrid decomposition:** ARIMA mean + Residual ML")
        aa, rr = hyb_parts
        fig2, ax2 = plt.subplots(figsize=(12,3.5))
        ax2.plot(aa.index, aa.values, label="ARIMA mean", linestyle="--")
        ax2.plot(rr.index, rr.values, label="Predicted residual", linestyle="-.")
        ax2.grid(alpha=.3); ax2.legend()
        st.pyplot(fig2)

# -----------------------------
# BACKTEST (one-step, last-12 months)
# -----------------------------
with tab_bt:
    st.subheader("Backtest (last 12 months)")
    if len(df_full) < 24:
        st.info("Not enough history to backtest.")
    else:
        h = 12
        train_bt = df_full["Close"].iloc[:-h]
        test_bt  = df_full["Close"].iloc[-h:]

        # ARIMA order from trained model
        order = arima_order_from(arima_model)
        try:
            arima_bt = ARIMA(train_bt, order=order).fit()
            arima_test_fc = arima_bt.forecast(steps=h)
            arima_test_fc.index = test_bt.index
            a_mae, a_rmse, a_mape = calc_metrics(test_bt.values, arima_test_fc.values)
        except Exception as e:
            st.error(f"ARIMA backtest failed: {e}")
            a_mae = a_rmse = a_mape = np.nan
            arima_test_fc = pd.Series(index=test_bt.index, dtype=float)

        # ML backtest (single shot on test features)
        if (ml_model is not None) and (scaler is not None):
            df_bt = ensure_features(df_full.copy(), feature_cols)
            test_bt_df = df_bt.iloc[-h:]
            Xb = test_bt_df[feature_cols].copy()
            Xb_scaled = scaler.transform(Xb)
            ml_test_fc = pd.Series(ml_model.predict(Xb_scaled), index=test_bt.index)
            m_mae, m_rmse, m_mape = calc_metrics(test_bt.values, ml_test_fc.values)
        else:
            ml_test_fc = None
            m_mae = m_rmse = m_mape = np.nan

        c1, c2, c3 = st.columns(3)
        with c1: kpi_card("ARIMA RMSE", a_rmse)
        with c2: kpi_card("ML RMSE",    m_rmse if not np.isnan(m_rmse) else 0.0)
        with c3: kpi_card("ARIMA MAPE (%)", a_mape)

        st.markdown("**Backtest curves**")
        fig, ax = plt.subplots(figsize=(12,4))
        ax.plot(train_bt.index, train_bt.values, color="gray", alpha=0.6, label="Train")
        ax.plot(test_bt.index,  test_bt.values,  color="black", label="Test")
        ax.plot(arima_test_fc.index, arima_test_fc.values, "--o", label="ARIMA fc")
        if ml_test_fc is not None:
            ax.plot(ml_test_fc.index, ml_test_fc.values, "--x", label="ML fc")
        ax.axvline(test_bt.index[0], color="gray", alpha=0.4)
        ax.grid(alpha=.3); ax.legend()
        st.pyplot(fig)

        if a_mape > 50:
            st.warning("ARIMA backtest MAPE is high — series is volatile and scale-sensitive; prefer RMSE/MAE.")

# -----------------------------
# RESIDUALS (ARIMA in-sample)
# -----------------------------
with tab_resid:
    st.subheader("ARIMA in-sample residuals")
    try:
        ins_pred = arima_model.predict(start=df_full.index[0], end=df_full.index[-1])
        resid = df_full["Close"].loc[ins_pred.index] - ins_pred
        fig, ax = plt.subplots(figsize=(12,3.8))
        ax.plot(resid.index, resid.values)
        ax.axhline(0, color="gray", alpha=.5)
        ax.set_title("Residuals (actual - in-sample prediction)")
        ax.grid(alpha=.3)
        st.pyplot(fig)

        st.markdown("**Residual ACF (quick look, 24 lags)**")
        fig2, ax2 = plt.subplots(figsize=(10,3.5))
        plot_acf(resid.dropna(), lags=24, ax=ax2)
        st.pyplot(fig2)
    except Exception as e:
        st.info(f"Residual plot not available: {e}")

# -----------------------------
# FEATURE IMPORTANCE (ML)
# -----------------------------
with tab_feat:
    st.subheader("Feature Importance (trend model)")
    if ml_model is None or scaler is None:
        st.info("ML model/scaler not found.")
    else:
        if hasattr(ml_model, "feature_importances_"):
            imp = ml_model.feature_importances_
            order = np.argsort(imp)[::-1]
            labels = [feature_cols[i] for i in order]
            vals = imp[order]
            fig, ax = plt.subplots(figsize=(7,4))
            ax.barh(labels[::-1], vals[::-1])
            ax.set_title("Feature importances")
            st.pyplot(fig)
        elif hasattr(ml_model, "coef_"):
            coef = np.ravel(ml_model.coef_)
            order = np.argsort(np.abs(coef))[::-1]
            labels = [feature_cols[i] for i in order]
            vals = coef[order]
            fig, ax = plt.subplots(figsize=(7,4))
            ax.barh(labels[::-1], vals[::-1])
            ax.set_title("Linear coefficients (magnitude)")
            st.pyplot(fig)
        else:
            st.info("Model does not expose importances or coefficients.")

# -----------------------------
# DATA EXPLORER
# -----------------------------
with tab_data:
    st.subheader("Data Explorer")
    st.dataframe(df_full.tail(20))
    st.markdown("**Summary**")
    st.write(df_full.describe())

# -----------------------------
# DOWNLOAD CENTER
# -----------------------------
with tab_dl:
    st.subheader("Download Center")
    out = pd.DataFrame(index=arima_fc.index)
    out["ARIMA"] = arima_fc.values
    if ml_fc is not None: out["ML"] = ml_fc.values
    if hyb_fc is not None: out["Hybrid"] = hyb_fc.values
    out_csv = out.reset_index().rename(columns={"index":"Date"}).to_csv(index=False)
    st.download_button(
        "Download forecast CSV",
        data=out_csv,
        file_name=f"forecast_{horizon}m.csv",
        mime="text/csv"
    )

    # Also let user download the engineered latest frame used by ML (for transparency)
    df_ml_csv = df_ml.reset_index().rename(columns={"index":"Date"}).to_csv(index=False)
    st.download_button(
        "Download engineered data (for ML)",
        data=df_ml_csv,
        file_name="engineered_data.csv",
        mime="text/csv"
    )

# st.caption("Tip: If values ever differ from your Colab run, ensure the same files are in the app folder: "
#            "`data.csv`, `arima_model.pkl`, `best_ml_model.pkl`, `rf_resid.pkl`, `scaler.pkl`, "
#            "`scaler_resid.pkl`, and either `features.json` or `feature_cols.json`.")
