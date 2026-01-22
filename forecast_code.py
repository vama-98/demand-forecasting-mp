"""
SALES FORECASTING DASHBOARD v2
(Adstocked Spend + Recursive Forecasting + Baseline Auto-Suggest + Calibration)

Run:
    streamlit run forecast_app_v2.py

What this version fixes/improves:
- ✅ Auto-suggest baseline discounts per SKU × Channel using LAST 30 DAYS MEDIAN of AVAILABLE history
  (still fully editable/overrideable in UI)
- ✅ Uses available-history max date (NOT pd.Timestamp.now()) for lookbacks → no empty-window surprises
- ✅ Avoids feature KeyErrors by auto-adding missing columns before model.predict
- ✅ Faster training defaults (fewer trees, sensible params, fewer CPU threads to avoid Windows thrash)
- ✅ Works with xgboost==3.1.3 (no early_stopping_rounds / eval_metric in fit)
- ✅ Adds simple cyclic calendar encodings (sin/cos) → usually improves weekday/seasonality learning
- ✅ Calibration factor (last 30 days) + log-space confidence bands

Notes:
- Shopify selection logic is LEFT AS-IS (per your request). We assume D2C appears as Channel == "Shopify".
"""

import os
import pickle
import warnings
from datetime import timedelta

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

# =============================================================================
# CONFIG
# =============================================================================
# =============================================================================
# TRAINING METRICS OUTPUT
# =============================================================================
def save_training_metrics(metrics: dict,
                          path_txt: str = "training_metrics.txt",
                          path_json: str = "training_metrics.json") -> bool:
    """Save training metrics to both TXT and JSON."""
    try:
        # TXT (human-readable)
        lines = []
        for k, v in metrics.items():
            if isinstance(v, float):
                lines.append(f"{k}: {v:.6f}")
            else:
                lines.append(f"{k}: {v}")
        with open(path_txt, "w", encoding="utf-8") as f:
            f.write("\n".join(lines) + "\n")

        # JSON (structured)
        import json
        with open(path_json, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2, default=str)

        return True
    except Exception:
        return False



st.set_page_config(page_title="Sales Forecasting", layout="wide", page_icon="📊")

SALES_FILE = "data/Sales_Data_2025.parquet"
SPENDS_FILE = "data/Spends_Data_2025.parquet"
OFFERS_FILE = "data/offer.xlsx"

UNITS_MODEL_FILE = "units_model.pkl"
FEATURE_COLS_FILE = "feature_columns.pkl"
CHANNEL_MAP_FILE = "channel_mapping.pkl"
TITLE_MAP_FILE = "title_mapping.pkl"
SIGMA_FILE = "model_sigmas.pkl"

ADSTOCK_ALPHA_DEFAULT = 0.30
CALIB_MIN = 0.70
CALIB_MAX = 1.30
import os

def _debug_file(path: str):
    #st.write(f"🔎 Checking: {path}")
    #st.write("exists:", os.path.exists(path))
    if os.path.exists(path):
        #st.write("size_bytes:", os.path.getsize(path))
        with open(path, "rb") as f:
            head = f.read(16)
        #st.write("first_16_bytes:", head)

# inside load_data(), before reading:
_debug_file(SALES_FILE)
_debug_file(SPENDS_FILE)

def load_offers_excel(path: str) -> pd.DataFrame:
    import pandas as pd

    def norm(x):
        if pd.isna(x):
            return ""
        s = str(x)
        s = s.replace("\ufeff", "").replace("\xa0", " ")
        s = " ".join(s.split())
        return s.strip().lower()

    xls = pd.ExcelFile(path, engine="openpyxl")
    #st.write("offer.xlsx sheets:", xls.sheet_names)  # safe to print

    best_df = pd.DataFrame()

    # Try each sheet
    for sheet in xls.sheet_names:
        raw = pd.read_excel(path, engine="openpyxl", sheet_name=sheet, header=None)

        # Search more rows + substring detection (handles merged cells/weird spacing)
        header_row_idx = None
        for i in range(min(50, len(raw))):
            row_vals = [norm(v) for v in raw.iloc[i].tolist()]
            blob = " | ".join([v for v in row_vals if v])  # joined row text

            has_title = "title" in blob
            has_offer = "offer" in blob
            has_discount = "discount" in blob

            if has_title and has_offer and has_discount:
                header_row_idx = i
                break

        if header_row_idx is None:
            continue

        df = pd.read_excel(path, engine="openpyxl", sheet_name=sheet, header=header_row_idx)

        # Clean column names
        df.columns = (
            df.columns.astype(str)
            .str.replace("\ufeff", "", regex=False)
            .str.replace("\xa0", " ", regex=False)
            .str.replace(r"\s+", " ", regex=True)
            .str.strip()
        )

        # If it has the expected columns, standardize and return immediately
        cols_norm = {c: c.strip().lower().replace(" ", "") for c in df.columns}
        title_col = next((c for c, n in cols_norm.items() if "title" == n), None)
        offer_col = next((c for c, n in cols_norm.items() if "offer" == n), None)
        disc_col  = next((c for c, n in cols_norm.items() if "discount" in n), None)

        if title_col and offer_col and disc_col:
            df = df.rename(columns={title_col: "Title", offer_col: "Offer", disc_col: "Discount"})
            return df

        # Keep best attempt (for debugging)
        best_df = df

    # If nothing matched, return whatever we got (or empty)
    return best_df if not best_df.empty else pd.DataFrame()


# =============================================================================
# DATA LOADING
# =============================================================================

@st.cache_data(show_spinner=False)
def load_data():
    try:
        sales_df = pd.read_parquet(SALES_FILE)
        sales_df.columns = sales_df.columns.str.strip()
        sales_df["Order_date"] = pd.to_datetime(sales_df["Order_date"], dayfirst=True, errors="coerce")

        spends_df = pd.read_parquet(SPENDS_FILE)
        spends_df.columns = spends_df.columns.str.strip()
        spends_df["date_start"] = pd.to_datetime(spends_df["date_start"], errors="coerce")
        spends_df.rename(columns={"Master_title": "Master_Title"}, inplace=True)
        offers_df = load_offers_excel(OFFERS_FILE)
        # Remove Offline
        if "source_channel" in sales_df.columns:
            sales_df = sales_df[sales_df["source_channel"] != "Offline"].copy()

        sales_df = sales_df.dropna(subset=["Order_date", "Master_Title", "Channel"])
        spends_df = spends_df.dropna(subset=["date_start", "Master_Title"])

        return sales_df, spends_df, offers_df
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None


# =============================================================================
# HELPERS
# =============================================================================
def estimate_spend_elasticity(
    daily_data: pd.DataFrame,
    product: str,
    channel: str,
    days: int = 120,
    fallback: float = 0.15,
) -> float:
    """
    Estimate elasticity from recent history:
      log(Units) ~ a + e * log(Spend_Adstock + 1)
    """
    d = daily_data[(daily_data["Master_Title"] == product) & (daily_data["Channel"] == channel)].copy()
    if d.empty or ("Spend_Adstock" not in d.columns):
        return float(fallback)

    d = d.sort_values("Order_date")
    cutoff = d["Order_date"].max() - pd.Timedelta(days=days)
    d = d[d["Order_date"] >= cutoff]

    d = d.dropna(subset=["Units_Sold", "Spend_Adstock"])
    d = d[(d["Units_Sold"] > 0) & (d["Spend_Adstock"] >= 0)]

    if len(d) < 45:
        return float(fallback)

    x = np.log1p(d["Spend_Adstock"].astype(float).values)
    y = np.log(d["Units_Sold"].astype(float).values)

    try:
        e = float(np.polyfit(x, y, 1)[0])
    except Exception:
        e = float(fallback)

    return float(np.clip(e, 0.05, 0.30))

def show_feature_importance(units_model, feature_cols, top_n=40):
    import pandas as pd

    booster = units_model.get_booster()
    score = booster.get_score(importance_type="gain")  # dict: {feature_name: gain}

    rows = []
    for k, v in score.items():
        # Case 1: keys look like "f12"
        if isinstance(k, str) and k.startswith("f") and k[1:].isdigit():
            idx = int(k[1:])
            name = feature_cols[idx] if idx < len(feature_cols) else k
        else:
            # Case 2: keys are already feature names
            name = k
        rows.append((name, float(v)))

    imp_df = pd.DataFrame(rows, columns=["Feature", "Gain"]).sort_values("Gain", ascending=False)

    # Ensure Shopify features show up even if missing
    for f in ["Shopify_Discount", "Shopify_Units", "Discount_Gap"]:
        if f not in imp_df["Feature"].values:
            imp_df = pd.concat([imp_df, pd.DataFrame([[f, 0.0]], columns=["Feature", "Gain"])], ignore_index=True)

    # Aggregate duplicates (sometimes both f-keys and names can map to same feature)
    imp_df = imp_df.groupby("Feature", as_index=False)["Gain"].sum().sort_values("Gain", ascending=False)

    return imp_df.head(top_n), imp_df


def load_training_metrics(path: str = "training_metrics.json") -> dict | None:
    import json, os
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None

def _ensure_features(df: pd.DataFrame, feature_cols: list[str]) -> pd.DataFrame:
    """Ensure df has all feature_cols; add missing with 0.0, then return ordered frame."""
    out = df.copy()
    for c in feature_cols:
        if c not in out.columns:
            out[c] = 0.0
    return out[feature_cols]


def _cyclic_encode(series: pd.Series, period: int, prefix: str) -> pd.DataFrame:
    x = series.astype(float)
    return pd.DataFrame({
        f"{prefix}_sin": np.sin(2 * np.pi * x / period),
        f"{prefix}_cos": np.cos(2 * np.pi * x / period),
    })


def _adstock_ewm(s: pd.Series, alpha: float) -> pd.Series:
    return s.ewm(alpha=alpha, adjust=False).mean()


def _adstock_next(prev_adstock: float, spend_today: float, alpha: float) -> float:
    return alpha * float(spend_today) + (1.0 - alpha) * float(prev_adstock)


def _compute_spend_dow_multiplier(daily_data: pd.DataFrame, product: str) -> dict:
    d = daily_data[daily_data["Master_Title"] == product][["Order_date", "Spend"]].copy()
    if d.empty:
        return {i: 1.0 for i in range(7)}
    d["dow"] = d["Order_date"].dt.dayofweek
    m = d.groupby("dow")["Spend"].mean()
    if m.empty or float(m.mean()) == 0.0:
        return {i: 1.0 for i in range(7)}
    m = m / float(m.mean())
    out = {int(k): float(v) for k, v in m.to_dict().items()}
    for i in range(7):
        out.setdefault(i, 1.0)
    return out


# =============================================================================
# AUTO-SUGGEST BASELINE DISCOUNTS
# =============================================================================

def suggested_baseline_discounts(
    daily_data: pd.DataFrame,
    sku: str,
    channel: str,
    lookback_days: int = 30,
    shopify_channel_name: str = "Shopify",
):
    """Returns (channel_baseline, shopify_baseline).

    Uses last N days relative to AVAILABLE HISTORY (max date in each slice), not pd.Timestamp.now().
    Shopify logic left as-is: Channel == "Shopify".
    """

    def _median_lastN(df_slice: pd.DataFrame) -> float | None:
        if df_slice.empty:
            return None
        df_slice = df_slice.dropna(subset=["Order_date", "Discount_Pct"]).copy()
        if df_slice.empty:
            return None
        max_dt = df_slice["Order_date"].max()
        cutoff = max_dt - pd.Timedelta(days=lookback_days)
        recent = df_slice[df_slice["Order_date"] >= cutoff]
        if recent.empty:
            return None
        return float(recent["Discount_Pct"].median())

    # SKU × selected channel
    slice_ch = daily_data[(daily_data["Master_Title"] == sku) & (daily_data["Channel"] == channel)].copy()
    ch_med = _median_lastN(slice_ch)

    # SKU × Shopify channel (assumption: Channel value is "Shopify")
    slice_shop = daily_data[(daily_data["Master_Title"] == sku) & (daily_data["Channel"] == shopify_channel_name)].copy()
    shop_med = _median_lastN(slice_shop)

    # Fallbacks
    if ch_med is None:
        ch_med = float(slice_ch["Discount_Pct"].median()) if not slice_ch.empty else 20.0
    if shop_med is None:
        shop_med = float(slice_shop["Discount_Pct"].median()) if not slice_shop.empty else 35.0

    return ch_med, shop_med


# =============================================================================
# FEATURE ENGINEERING
# =============================================================================

@st.cache_data(show_spinner=False)
def prepare_training_data(sales_df: pd.DataFrame, spends_df: pd.DataFrame, adstock_alpha: float = ADSTOCK_ALPHA_DEFAULT) -> pd.DataFrame:
    df = sales_df.copy()

    # Safe discount + ASP
    df["ASP"] = np.where(df["Units_Sold"] > 0, df["Gross_Sales"] / df["Units_Sold"], 0)
    df["Discount_Pct"] = np.where(df["GMV"] > 0, ((df["GMV"] - df["Gross_Sales"]) / df["GMV"]) * 100, 0)

    daily_sales = df.groupby(["Order_date", "Master_Title", "Channel"], as_index=False).agg({
        "Units_Sold": "sum",
        "Gross_Sales": "sum",
        "GMV": "sum",
        "Discount_Pct": "mean",
        "source_channel": "first" if "source_channel" in df.columns else "size",
    })

    daily_sales["ASP"] = np.where(daily_sales["Units_Sold"] > 0, daily_sales["Gross_Sales"] / daily_sales["Units_Sold"], 0)
    daily_sales["ASP"] = daily_sales["ASP"].replace([np.inf, -np.inf], 0).fillna(0)

    daily_spends = spends_df.groupby(["date_start", "Master_Title"], as_index=False).agg({"Spend": "sum"})

    daily_data = daily_sales.merge(
        daily_spends,
        left_on=["Order_date", "Master_Title"],
        right_on=["date_start", "Master_Title"],
        how="left",
    )
    daily_data["Spend"] = daily_data["Spend"].fillna(0)
    daily_data = daily_data.drop(columns=["date_start"])

    # Time features
    daily_data["Day_of_Week"] = daily_data["Order_date"].dt.dayofweek
    daily_data["Day_of_Month"] = daily_data["Order_date"].dt.day
    daily_data["Month"] = daily_data["Order_date"].dt.month
    daily_data["Week_of_Year"] = daily_data["Order_date"].dt.isocalendar().week.astype(int)
    daily_data["Is_Weekend"] = (daily_data["Day_of_Week"] >= 5).astype(int)

    # Practical regime flags
    daily_data["Is_Month_Start"] = (daily_data["Day_of_Month"] <= 3).astype(int)
    daily_data["Is_Month_End"] = (daily_data["Day_of_Month"] >= 28).astype(int)
    daily_data["Is_Payday_Window"] = daily_data["Day_of_Month"].isin([28, 29, 30, 31, 1, 2, 3]).astype(int)

    # Sort and group
    daily_data = daily_data.sort_values(["Master_Title", "Channel", "Order_date"]).reset_index(drop=True)
    g = daily_data.groupby(["Master_Title", "Channel"], sort=False)

    # Lags
    daily_data["Units_Lag_1"] = g["Units_Sold"].shift(1)
    daily_data["Units_Lag_7"] = g["Units_Sold"].shift(7)

    # Rolling (past-only, no leakage)
    daily_data["Units_Rolling_7"] = g["Units_Sold"].transform(lambda s: s.shift(1).rolling(7, min_periods=1).mean())
    daily_data["Units_Rolling_30"] = g["Units_Sold"].transform(lambda s: s.shift(1).rolling(30, min_periods=1).mean())

    # Adstock + saturation
    daily_data["Spend_Adstock"] = g["Spend"].transform(lambda s: _adstock_ewm(s, adstock_alpha))
    daily_data["Spend_Sat"] = np.log1p(daily_data["Spend_Adstock"])

    # Cross-channel proxy (Shopify)
    shopify_data = daily_data[daily_data["Channel"] == "Shopify"][["Order_date", "Master_Title", "Discount_Pct", "Units_Sold"]].copy()
    shopify_data.rename(columns={"Discount_Pct": "Shopify_Discount", "Units_Sold": "Shopify_Units"}, inplace=True)

    daily_data = daily_data.merge(shopify_data, on=["Order_date", "Master_Title"], how="left")
    daily_data["Shopify_Discount"] = daily_data["Shopify_Discount"].fillna(0)
    daily_data["Shopify_Units"] = daily_data["Shopify_Units"].fillna(0)
    daily_data["Discount_Gap"] = daily_data["Discount_Pct"] - daily_data["Shopify_Discount"]

    # Fill NaNs for model
    fill0 = [
        "Units_Lag_1", "Units_Lag_7", "Units_Rolling_7", "Units_Rolling_30",
        "Spend_Adstock", "Spend_Sat",
        "Shopify_Discount", "Shopify_Units", "Discount_Gap",
    ]
    for c in fill0:
        if c in daily_data.columns:
            daily_data[c] = daily_data[c].fillna(0)

    # Filter valid rows
    daily_data = daily_data[(daily_data["Units_Sold"] > 0) & (daily_data["Gross_Sales"] >= 0)].copy()

    return daily_data


# =============================================================================
# MODEL TRAINING
# =============================================================================

def train_models(daily_data: pd.DataFrame):
    st.write("### Training model...")

    # Fewer features, more stable
    base_features = [
        "Discount_Pct",
        "Spend_Adstock",
        "Spend_Sat",
        "ASP",
        "Units_Lag_1",
        "Units_Lag_7",
        "Units_Rolling_7",
        "Units_Rolling_30",
        "Shopify_Discount",
        "Shopify_Units",
        "Discount_Gap",
        "Is_Weekend",
        "Is_Month_Start",
        "Is_Month_End",
        "Is_Payday_Window",
    ]

    d = daily_data.copy()

    # Encodings
    d["Channel"] = d["Channel"].astype("category")
    d["Master_Title"] = d["Master_Title"].astype("category")
    d["Channel_Encoded"] = d["Channel"].cat.codes
    d["Title_Encoded"] = d["Master_Title"].cat.codes

    # Cyclic encodings
    dow_cyc = _cyclic_encode(d["Day_of_Week"], 7, "dow")
    mon_cyc = _cyclic_encode(d["Month"], 12, "mon")
    woy_cyc = _cyclic_encode(d["Week_of_Year"], 52, "woy")
    d = pd.concat([d, dow_cyc, mon_cyc, woy_cyc], axis=1)

    feature_cols = base_features + [
        "Channel_Encoded", "Title_Encoded",
        "dow_sin", "dow_cos", "mon_sin", "mon_cos", "woy_sin", "woy_cos",
    ]

    d = d.sort_values("Order_date").reset_index(drop=True)

    X = d[feature_cols].astype(float)
    y = np.log1p(d["Units_Sold"].astype(float).values)

    cut = int(len(d) * 0.80)
    X_train, X_test = X.iloc[:cut], X.iloc[cut:]
    y_train, y_test = y[:cut], y[cut:]

    # Faster + stable params
    params = dict(
        n_estimators=900,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        reg_lambda=2.0,
        min_child_weight=5,
        tree_method="hist",
        max_bin=256,
        n_jobs=max(1, (os.cpu_count() or 4) // 2),
        random_state=42,
    )

    units_model = XGBRegressor(**params)

    # xgboost==3.1.3: no early_stopping_rounds / eval_metric in fit
    # Put eval_metric in constructor instead.
    units_model.set_params(eval_metric="rmse")

    st.info(f"Training on {len(d):,} rows with {len(feature_cols)} features...")
    units_model.fit(X_train, y_train)

    pred_log = units_model.predict(X_test)
    pred = np.expm1(pred_log)
    true = np.expm1(y_test)

    mae = float(np.mean(np.abs(true - pred)))
    mape = float(np.mean(np.abs(true - pred) / np.clip(true, 1, None)))
    wape = float(np.sum(np.abs(true - pred)) / np.clip(np.sum(np.abs(true)), 1.0, None))
    bias = float(np.sum(true - pred) / np.clip(np.sum(np.abs(true)), 1.0, None))
    
    st.success(f"✅ Units MAE: {mae:,.1f} | MAPE: {mape*100:.1f}%")

    sigma_units_log = float(np.std(y_test - pred_log))
    metrics_payload = {
    "rows": int(len(d)),
    "features": int(len(feature_cols)),
    "mae_units": float(mae),
    "mape_units": float(mape),
    "wape_units": float(wape),
    "sigma_units_log": float(sigma_units_log),
    "train_end_date": str(d["Order_date"].iloc[cut-1].date()) if len(d) > 0 and cut > 0 else "",
    "test_start_date": str(d["Order_date"].iloc[cut].date()) if len(d) > 0 and cut < len(d) else "",
    }

    if save_training_metrics(metrics_payload):
        st.info("🧾 Saved training_metrics.txt and training_metrics.json in your script folder")
    else:
        st.warning("Could not write training metrics files (check folder permissions).")

    # Save
    with open(UNITS_MODEL_FILE, "wb") as f:
        pickle.dump(units_model, f)
    with open(FEATURE_COLS_FILE, "wb") as f:
        pickle.dump(feature_cols, f)

    channel_to_code = {cat: int(i) for i, cat in enumerate(d["Channel"].cat.categories)}
    title_to_code = {cat: int(i) for i, cat in enumerate(d["Master_Title"].cat.categories)}
    with open(CHANNEL_MAP_FILE, "wb") as f:
        pickle.dump(channel_to_code, f)
    with open(TITLE_MAP_FILE, "wb") as f:
        pickle.dump(title_to_code, f)
    with open(SIGMA_FILE, "wb") as f:
        pickle.dump({"sigma_units_log": sigma_units_log}, f)

    return units_model, feature_cols, channel_to_code, title_to_code, sigma_units_log


def load_models():
    try:
        with open(UNITS_MODEL_FILE, "rb") as f:
            units_model = pickle.load(f)
        with open(FEATURE_COLS_FILE, "rb") as f:
            feature_cols = pickle.load(f)
        with open(CHANNEL_MAP_FILE, "rb") as f:
            channel_to_code = pickle.load(f)
        with open(TITLE_MAP_FILE, "rb") as f:
            title_to_code = pickle.load(f)

        sigma_units_log = 0.0
        if os.path.exists(SIGMA_FILE):
            with open(SIGMA_FILE, "rb") as f:
                sigma_units_log = float(pickle.load(f).get("sigma_units_log", 0.0))

        return units_model, feature_cols, channel_to_code, title_to_code, sigma_units_log
    except Exception:
        return None, None, None, None, 0.0


# =============================================================================
# CALIBRATION + BANDS
# =============================================================================

def calibration_factor_last30(
    daily_data: pd.DataFrame,
    units_model,
    feature_cols,
    product: str,
    channel: str,
    channel_to_code: dict,
    title_to_code: dict,
    days: int = 30,
):
    d = daily_data[(daily_data["Master_Title"] == product) & (daily_data["Channel"] == channel)].copy()
    if d.empty:
        return 1.0

    d = d.sort_values("Order_date")
    cutoff = d["Order_date"].max() - pd.Timedelta(days=days)
    d = d[d["Order_date"] >= cutoff]
    if len(d) < 14:
        return 1.0

    # Add encodings and cyclics (must match training)
    d["Channel_Encoded"] = int(channel_to_code.get(channel, 0))
    d["Title_Encoded"] = int(title_to_code.get(product, 0))

    d = pd.concat([
        d,
        _cyclic_encode(d["Day_of_Week"], 7, "dow"),
        _cyclic_encode(d["Month"], 12, "mon"),
        _cyclic_encode(d["Week_of_Year"], 52, "woy"),
    ], axis=1)

    X = _ensure_features(d, feature_cols).astype(float)
    y = d["Units_Sold"].values.astype(float)

    pred_log = units_model.predict(X)
    pred = np.expm1(pred_log)

    actual_sum = float(np.sum(y))
    pred_sum = float(np.sum(pred))
    if pred_sum <= 0:
        return 1.0

    f = actual_sum / pred_sum
    return float(np.clip(f, CALIB_MIN, CALIB_MAX))


def add_confidence_bands_logspace(df: pd.DataFrame, sigma_log: float, z: float = 1.64):
    if df is None or df.empty or sigma_log <= 0:
        return df

    out = df.copy()
    if "Predicted_Log_Units" not in out.columns:
        out["Predicted_Log_Units"] = np.log1p(np.clip(out["Predicted_Units"].values, 0, None))

    out["Lower"] = np.expm1(out["Predicted_Log_Units"] - z * sigma_log)
    out["Upper"] = np.expm1(out["Predicted_Log_Units"] + z * sigma_log)
    out["Lower"] = np.clip(out["Lower"], 0, None)
    out["Upper"] = np.clip(out["Upper"], 0, None)
    return out


# =============================================================================
# RECURSIVE FORECAST
# =============================================================================
def _compute_units_dow_multiplier(daily_data: pd.DataFrame, product: str, channel: str) -> dict:
    d = daily_data[(daily_data["Master_Title"] == product) & (daily_data["Channel"] == channel)][
        ["Order_date", "Units_Sold"]
    ].copy()
    if d.empty:
        return {i: 1.0 for i in range(7)}
    d["dow"] = d["Order_date"].dt.dayofweek
    m = d.groupby("dow")["Units_Sold"].mean()
    if m.empty or float(m.mean()) == 0.0:
        return {i: 1.0 for i in range(7)}
    m = m / float(m.mean())
    out = {int(k): float(v) for k, v in m.to_dict().items()}
    for i in range(7):
        out.setdefault(i, 1.0)
    return out
def generate_forecast_recursive(
    product: str,
    channel: str,
    start_date,
    end_date,
    shopify_offers,
    target_sales,
    baseline_shopify_discount: float,
    baseline_target_discount: float,
    monthly_spend: float,
    daily_data: pd.DataFrame,
    units_model,
    feature_cols,
    channel_to_code: dict,
    title_to_code: dict,
    adstock_alpha: float = 0.30,
    max_daily_change_pct: float = 0.15,
    stabilization_mode: str = "adaptive", 
    shopify_units_override: pd.Series | None = None,
):
    # --- 1. SETUP ---
    dates = pd.date_range(start=pd.to_datetime(start_date), end=pd.to_datetime(end_date), freq="D")
    fc = pd.DataFrame({"Date": dates})
    hist = daily_data[(daily_data["Master_Title"] == product) & (daily_data["Channel"] == channel)].sort_values("Order_date")
    
    if hist.empty: 
        return None

    # Standard Calendar Features
    fc["Day_of_Week"] = fc["Date"].dt.dayofweek
    fc["Day_of_Month"] = fc["Date"].dt.day
    fc["Month"] = fc["Date"].dt.month
    fc["Week_of_Year"] = fc["Date"].dt.isocalendar().week.astype(int)
    fc["Is_Weekend"] = (fc["Day_of_Week"] >= 5).astype(int)
    fc["Is_Month_Start"] = (fc["Day_of_Month"] <= 3).astype(int)
    fc["Is_Month_End"] = (fc["Day_of_Month"] >= 28).astype(int)
    fc["Is_Payday_Window"] = fc["Day_of_Month"].isin([28, 29, 30, 31, 1, 2, 3]).astype(int)

    # --- 2. EXTERNAL IMPACTS ---
    num_days = len(fc)
    daily_spend_base = float(monthly_spend) / num_days if num_days else 0.0
    dow_mult = _compute_spend_dow_multiplier(daily_data, product)
    fc["Spend"] = fc["Day_of_Week"].map(lambda d: dow_mult.get(int(d), 1.0)) * daily_spend_base
    
    fc["Discount_Pct"] = float(baseline_target_discount)
    fc["Shopify_Discount"] = float(baseline_shopify_discount)
    fc["Is_Promo_Event"] = 0
    
    for offer in shopify_offers:
        mask = fc["Date"].isin(pd.to_datetime(offer.get("dates", [])))
        fc.loc[mask, "Shopify_Discount"] = float(offer.get("discount", baseline_shopify_discount))
        fc.loc[mask, "Is_Promo_Event"] = 1
    
    for sale in target_sales:
        mask = fc["Date"].isin(pd.to_datetime(sale.get("dates", [])))
        fc.loc[mask, "Discount_Pct"] = float(sale.get("discount", baseline_target_discount))
        fc.loc[mask, "Is_Promo_Event"] = 1

    # --- 3. MODEL PREP ---
    fc["Channel_Encoded"] = int(channel_to_code.get(channel, 0))
    fc["Title_Encoded"] = int(title_to_code.get(product, 0))
    fc = pd.concat([fc, _cyclic_encode(fc["Day_of_Week"], 7, "dow"), 
                    _cyclic_encode(fc["Month"], 12, "mon"), 
                    _cyclic_encode(fc["Week_of_Year"], 52, "woy")], axis=1)

    # --- 4. IMPROVED RECURSIVE STATE ---
    hist_units = hist["Units_Sold"].tolist()
    
    # Stronger baseline anchors
    recent_median = float(hist["Units_Sold"].tail(30).median())
    recent_mean = float(hist["Units_Sold"].tail(30).mean())
    fixed_baseline = max(recent_median, 1.0)
    
    # Conservative initial rolling values
    roll7 = float(np.median(hist_units[-7:])) if len(hist_units) >= 7 else recent_median
    roll30 = float(np.median(hist_units[-30:])) if len(hist_units) >= 30 else recent_median
    prev_adstock = float(hist["Spend_Adstock"].iloc[-1]) if "Spend_Adstock" in hist.columns else 0.0

    # Track prediction history for stabilization
    prediction_buffer = []

    # --- 5. IMPROVED RECURSIVE LOOP ---
    preds = []
    
    for i in range(len(fc)):
        # Adstock
        prev_adstock = _adstock_next(prev_adstock, float(fc.loc[i, "Spend"]), adstock_alpha)
        fc.loc[i, "Spend_Adstock"] = prev_adstock
        fc.loc[i, "Spend_Sat"] = np.log1p(prev_adstock)
        
        # Lag features - use HISTORICAL data for early predictions
        if len(prediction_buffer) < 7:
            # Use actual history for first week
            lag1 = float(hist_units[-1]) if hist_units else fixed_baseline
            lag7 = float(hist_units[-7]) if len(hist_units) >= 7 else lag1
        else:
            # Gradually transition to predictions
            lag1 = prediction_buffer[-1]
            lag7 = prediction_buffer[-7] if len(prediction_buffer) >= 7 else lag1
        
        fc.loc[i, "Units_Lag_1"] = lag1
        fc.loc[i, "Units_Lag_7"] = lag7
        fc.loc[i, "Units_Rolling_7"] = roll7
        fc.loc[i, "Units_Rolling_30"] = roll30
        fc.loc[i, "Discount_Gap"] = float(fc.loc[i, "Discount_Pct"] - fc.loc[i, "Shopify_Discount"])
        
        # XGBoost Raw Prediction
        X_i = _ensure_features(fc.loc[[i]], feature_cols).astype(float)
        pred_raw = max(float(np.expm1(units_model.predict(X_i)[0])), 0.0)

        # IMPROVED: Multi-stage stabilization
        is_promo = fc.loc[i, "Is_Promo_Event"] == 1
        
        # Stage 1: Blend with lag (less aggressive)
        pred_step1 = (0.75 * pred_raw) + (0.25 * lag1)
        
        # Stage 2: Anchor to baseline (stronger pull)
        baseline_weight = 0.4 if is_promo else 0.5  # Less anchor during promos
        pred_step2 = (1 - baseline_weight) * pred_step1 + baseline_weight * fixed_baseline
        
        # Stage 3: Apply realistic caps based on historical volatility
        hist_std = float(hist["Units_Sold"].tail(60).std())
        hist_max = float(hist["Units_Sold"].tail(60).quantile(0.95))
        
        if is_promo:
            ceiling = min(hist_max * 2.0, fixed_baseline * 3.5)
        else:
            ceiling = min(hist_max * 1.2, fixed_baseline * 1.5)
        
        floor = max(fixed_baseline * 0.3, recent_median * 0.5)
        
        pred = np.clip(pred_step2, floor, ceiling)
        
        # Stage 4: Smooth out erratic jumps
        if len(prediction_buffer) > 0:
            max_change = prediction_buffer[-1] * (1 + max_daily_change_pct)
            min_change = prediction_buffer[-1] * (1 - max_daily_change_pct)
            pred = np.clip(pred, min_change, max_change)

        # Store prediction
        preds.append(pred)
        prediction_buffer.append(pred)
        
        # Update rolling windows SLOWLY to prevent drift
        roll7 = (0.90 * roll7) + (0.10 * pred)
        roll30 = (0.97 * roll30) + (0.03 * pred)

    # --- 6. FINALIZE ---
    fc["Predicted_Units"] = np.array(preds)
    fc["Predicted_Revenue"] = fc["Predicted_Units"] * (float(hist["ASP"].mean()) if "ASP" in hist.columns else 0.0)
    fc["Predicted_Log_Units"] = np.log1p(fc["Predicted_Units"])
    
    return fc
# =============================================================================
# PLOT
# =============================================================================

def plot_history_vs_forecast(hist_plot, baseline_fc, scenario_fc, forecast_start, show_baseline_trend=True, show_bands=True):
    if isinstance(forecast_start, pd.Timestamp):
        forecast_start = forecast_start.to_pydatetime()

    fig = go.Figure()

    # Historical actual
    fig.add_trace(go.Scatter(
        x=hist_plot["Order_date"],
        y=hist_plot["Units_Sold"],
        mode="lines",
        name="Historical (Actual)",
        line=dict(color="rgb(31, 119, 180)", width=1.5),
    ))

    # Show 30-day median baseline instead of 7DMA
    if show_baseline_trend and "Units_30D_Median" in hist_plot.columns:
        fig.add_trace(go.Scatter(
            x=hist_plot["Order_date"],
            y=hist_plot["Units_30D_Median"],
            mode="lines",
            name="Historical (30D Median)",
            line=dict(width=2, dash="dot", color="rgba(100, 149, 237, 0.6)"),
        ))

    # Add horizontal baseline reference line
    if not hist_plot.empty:
        recent_baseline = float(hist_plot["Units_Sold"].tail(30).median())
        fig.add_shape(
            type="line",
            x0=hist_plot["Order_date"].min(),
            x1=hist_plot["Order_date"].max(),
            y0=recent_baseline,
            y1=recent_baseline,
            line=dict(color="rgba(150, 150, 150, 0.3)", width=1, dash="dash"),
        )
        fig.add_annotation(
            x=hist_plot["Order_date"].max(),
            y=recent_baseline,
            text=f"30D Baseline: {recent_baseline:.0f}",
            showarrow=False,
            xanchor="right",
            yanchor="bottom",
            font=dict(size=10, color="gray"),
        )

    # Baseline forecast
    if baseline_fc is not None and not baseline_fc.empty:
        fig.add_trace(go.Scatter(
            x=baseline_fc["Date"],
            y=baseline_fc["Predicted_Units"],
            mode="lines",
            name="Baseline Forecast",
            line=dict(width=2, dash="dash", color="rgb(255, 127, 14)"),
        ))

    # Confidence bands
    if show_bands and scenario_fc is not None and ("Lower" in scenario_fc.columns) and ("Upper" in scenario_fc.columns):
        fig.add_trace(go.Scatter(
            x=scenario_fc["Date"],
            y=scenario_fc["Upper"],
            mode="lines",
            line=dict(width=0),
            showlegend=False,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=scenario_fc["Date"],
            y=scenario_fc["Lower"],
            mode="lines",
            fill="tonexty",
            fillcolor="rgba(44, 160, 101, 0.2)",
            line=dict(width=0),
            name="Confidence Band",
            hoverinfo="skip",
        ))

    # Scenario forecast
    if scenario_fc is not None and not scenario_fc.empty:
        fig.add_trace(go.Scatter(
            x=scenario_fc["Date"],
            y=scenario_fc["Predicted_Units"],
            mode="lines+markers",
            name="Scenario Forecast",
            marker=dict(size=5, color="rgb(44, 160, 101)"),
            line=dict(width=2.5, color="rgb(44, 160, 101)"),
        ))

    # Forecast start line
    fig.add_shape(
        type="line",
        x0=forecast_start,
        x1=forecast_start,
        y0=0,
        y1=1,
        xref="x",
        yref="paper",
        line=dict(width=2, dash="dash", color="gray"),
    )
    fig.add_annotation(
        x=forecast_start,
        y=1,
        xref="x",
        yref="paper",
        text="Forecast Start",
        showarrow=False,
        yanchor="bottom",
        xanchor="left",
    )

    fig.update_layout(
        title="Historical vs Forecast (Baseline vs Scenario)",
        xaxis_title="Date",
        yaxis_title="Units",
        hovermode="x unified",
        height=520,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )

    return fig


# =============================================================================
# APP
# =============================================================================

def main():
    st.title("📊 Sales Forecasting Dashboard")
    st.markdown("---")

    sales_df, spends_df, offers_df = load_data()
    if sales_df is None:
        st.error("Failed to load data files. Please check file paths.")
        return

    with st.sidebar:
        st.header("🔧 Model")
        adstock_alpha = st.slider("Adstock alpha", 0.05, 0.80, ADSTOCK_ALPHA_DEFAULT, 0.05)
        max_daily_change_pct = st.slider("Max day-to-day change cap", 0.05, 0.60, 0.20, 0.05)

        models_exist = all([
            os.path.exists(UNITS_MODEL_FILE),
            os.path.exists(FEATURE_COLS_FILE),
            os.path.exists(CHANNEL_MAP_FILE),
            os.path.exists(TITLE_MAP_FILE),
            os.path.exists(SIGMA_FILE),
        ])

        daily_data = prepare_training_data(sales_df, spends_df, adstock_alpha=adstock_alpha)

        if models_exist:
            st.success("✅ Model found")
            if st.button("🔄 Retrain Model"):
                train_models(daily_data)
                st.rerun()
        else:
            st.warning("⚠️ No model found")
            if st.button("🚀 Train Model"):
                train_models(daily_data)
                st.rerun()

        st.markdown("---")
        show_baseline_trend = st.checkbox("Show 30D median baseline", value=True)
        show_bands = st.checkbox("Show confidence bands", value=True)
        band_level = st.selectbox("Band level", ["90%", "95%"], index=0)
        z = 1.64 if band_level == "90%" else 1.96

    units_model, feature_cols, channel_to_code, title_to_code, sigma_units_log = load_models()
    if units_model is None:
        st.warning("⚠️ Please train the model from the sidebar.")
        return

    # Selection
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("📦 Product & Channel")
        products = sorted(daily_data["Master_Title"].dropna().unique())
        channels = sorted(daily_data["Channel"].dropna().unique())
        selected_product = st.selectbox("Select Product", products)
        selected_channel = st.selectbox("Select Channel", channels)

    with col2:
        st.subheader("📅 Forecast Period")
        min_date = sales_df["Order_date"].max() + timedelta(days=1)
        max_date = min_date + timedelta(days=90)
        start_date = st.date_input("Start Date", value=min_date, min_value=min_date, max_value=max_date)
        end_date = st.date_input("End Date", value=min_date + timedelta(days=30), min_value=min_date, max_value=max_date)

    st.markdown("---")

    # Auto-suggest baselines (UI overrideable)
    suggested_target, suggested_shopify = suggested_baseline_discounts(
        daily_data=daily_data,
        sku=selected_product,
        channel=selected_channel,
        lookback_days=30,
        shopify_channel_name="Shopify",
    )

    col3, col4 = st.columns(2)
    with col3:
        st.subheader("💰 Baseline Discounts (auto-suggested)")
        st.caption(f"Suggested {selected_channel}: {suggested_target:.1f}% | Suggested Shopify: {suggested_shopify:.1f}%")

        baseline_target = st.number_input(
            f"{selected_channel} Baseline Discount %",
            value=float(suggested_target),
            min_value=0.0,
            max_value=100.0,
            step=0.5,
        )
        baseline_shopify = st.number_input(
            "Shopify Baseline Discount %",
            value=float(suggested_shopify),
            min_value=0.0,
            max_value=100.0,
            step=0.5,
        )

    with col4:
        st.subheader("💵 Marketing Spend")
        monthly_spend = st.number_input("Monthly Spend (₹)", value=5_000_000, min_value=0, step=100_000)

    st.markdown("---")

    # Offers UI
    st.subheader("🎁 Shopify Offers (Scenario)")
    if offers_df is None or offers_df.empty or "Title" not in offers_df.columns:
        st.warning("No usable offers found in offer.xlsx (missing Title/Offer/Discount headers).")
        product_offers = pd.DataFrame()
    else:
        product_offers = offers_df[offers_df["Title"].astype(str).str.strip() == str(selected_product).strip()]


    shopify_offers = []
    if not product_offers.empty:
        offer_choices = sorted(product_offers["Offer"].dropna().unique().tolist())
        offer_choices = offer_choices + ["Other"]  # ✅ allow manual discount

        num_shopify_offers = st.number_input(
            "Number of Shopify Offers",
            min_value=0,
            max_value=10,
            value=0
        )

        for i in range(int(num_shopify_offers)):
            a, b, c = st.columns([1.3, 1.0, 2.0])

            with a:
                offer_name = st.selectbox(
                    f"Offer {i+1}",
                    offer_choices,
                    key=f"shopify_offer_{i}"
                )

            with b:
                # ✅ No discount display. Only input if "Other" else silent auto-pick.
                if offer_name == "Other":
                    offer_discount = st.number_input(
                        "Discount %",
                        min_value=0.0,
                        max_value=100.0,
                        value=float(baseline_shopify),   # sensible default
                        step=0.5,
                        key=f"shopify_other_disc_{i}"
                    )
                else:
                    # pull from file (no display)
                    offer_discount = float(
                        product_offers.loc[product_offers["Offer"] == offer_name, "Discount"].iloc[0]
                    )

            with c:
                offer_dates = st.multiselect(
                    "Apply on dates",
                    options=pd.date_range(start_date, end_date).strftime("%Y-%m-%d").tolist(),
                    key=f"shopify_dates_{i}",
                )

            if offer_dates:
                shopify_offers.append({
                    "name": offer_name,
                    "discount": float(offer_discount),
                    "dates": offer_dates
                })
    else:
        st.info("No offers data available for this product")

    st.markdown("---")

    st.subheader(f"🏷️ {selected_channel} Sale Events (Scenario)")
    num_target_sales = st.number_input("Number of Sale Events", min_value=0, max_value=10, value=0)

    target_sales = []
    for i in range(num_target_sales):
        d1, d2 = st.columns(2)
        with d1:
            sale_discount = st.number_input(f"Sale {i+1} Discount %", value=30, min_value=0, max_value=100, key=f"target_disc_{i}")
        with d2:
            sale_dates = st.multiselect(
                "Apply on dates",
                options=pd.date_range(start_date, end_date).strftime("%Y-%m-%d").tolist(),
                key=f"target_dates_{i}",
            )
        if sale_dates:
            target_sales.append({"discount": float(sale_discount), "dates": sale_dates})

    #top_imp, full_imp = show_feature_importance(units_model, feature_cols, top_n=40)

    #st.subheader("🔍 Model Feature Importance (Gain)")
    #st.dataframe(top_imp, use_container_width=True, height=500)

    #st.subheader("🧾 Shopify Feature Importance (Gain)")
    #shop = full_imp[full_imp["Feature"].isin(["Shopify_Discount", "Shopify_Units", "Discount_Gap"])]
    #st.dataframe(shop, use_container_width=True)

    
    st.markdown("---")
    if st.button("🔮 Generate Forecast", type="primary", use_container_width=True):
        with st.spinner("Generating forecasts..."):
            scenario_fc = generate_forecast_recursive(
                product=selected_product,
                channel=selected_channel,
                start_date=start_date,
                end_date=end_date,
                shopify_offers=shopify_offers,
                target_sales=target_sales,
                baseline_shopify_discount=float(baseline_shopify),
                baseline_target_discount=float(baseline_target),
                monthly_spend=float(monthly_spend),
                daily_data=daily_data,
                units_model=units_model,
                feature_cols=feature_cols,
                channel_to_code=channel_to_code,
                title_to_code=title_to_code,
                adstock_alpha=float(adstock_alpha),
                max_daily_change_pct=float(max_daily_change_pct),
                stabilization_mode="light",
            )

            baseline_fc = generate_forecast_recursive(
                product=selected_product,
                channel=selected_channel,
                start_date=start_date,
                end_date=end_date,
                shopify_offers=[],
                target_sales=[],
                baseline_shopify_discount=float(baseline_shopify),
                baseline_target_discount=float(baseline_target),
                monthly_spend=float(monthly_spend),
                daily_data=daily_data,
                units_model=units_model,
                feature_cols=feature_cols,
                channel_to_code=channel_to_code,
                title_to_code=title_to_code,
                adstock_alpha=float(adstock_alpha),
                max_daily_change_pct=float(max_daily_change_pct),
                stabilization_mode="medium",
            )

            if scenario_fc is None or baseline_fc is None:
                st.error("No historical data found for the selected product/channel.")
                return

            # Calibration (apply to both)
            calib = calibration_factor_last30(
                daily_data=daily_data,
                units_model=units_model,
                feature_cols=feature_cols,
                product=selected_product,
                channel=selected_channel,
                channel_to_code=channel_to_code,
                title_to_code=title_to_code,
                days=30,
            )
            

            for df_fc in (scenario_fc, baseline_fc):
                df_fc["Predicted_Units"] *= calib
                df_fc["Predicted_Revenue"] *= calib
                df_fc["Predicted_Log_Units"] = np.log1p(np.clip(df_fc["Predicted_Units"].values, 0, None))

            # Display Model Performance Metrics
            st.markdown("---")
            st.subheader("📊 Model Performance Metrics")

            metrics = load_training_metrics("training_metrics.json")
            if metrics:
                col_m1, col_m2, col_m3, col_m4 = st.columns(4)
                
                with col_m1:
                    st.metric(
                        "MAE (Units)", 
                        f"{metrics.get('mae_units', 0):,.1f}",
                        help="Mean Absolute Error on test set"
                    )
                
                with col_m2:
                    st.metric(
                        "MAPE", 
                        f"{metrics.get('mape_units', 0) * 100:.2f}%",
                        help="Mean Absolute Percentage Error"
                    )
                
                with col_m3:
                    st.metric(
                        "WAPE", 
                        f"{metrics.get('wape_units', 0) * 100:.2f}%",
                        help="Weighted Absolute Percentage Error"
                    )
                
                with col_m4:
                    bias_val = metrics.get('bias', 0)
                    st.metric(
                        "Bias", 
                        f"{bias_val * 100:.2f}%",
                        delta=f"{'Over' if bias_val > 0 else 'Under'}-forecasting" if abs(bias_val) > 0.01 else "Balanced",
                        help="Forecast bias (positive = over-forecasting, negative = under-forecasting)"
                    )
                
                # Optional: Show training details in expander
                with st.expander("📋 Training Details"):
                    st.write(f"**Training rows:** {metrics.get('rows', 'N/A'):,}")
                    st.write(f"**Features used:** {metrics.get('features', 'N/A')}")
                    st.write(f"**Train end date:** {metrics.get('train_end_date', 'N/A')}")
                    st.write(f"**Test start date:** {metrics.get('test_start_date', 'N/A')}")
                    st.write(f"**Sigma (log-space):** {metrics.get('sigma_units_log', 0):.4f}")
            else:
                st.warning("⚠️ Training metrics not found. Please retrain the model to see performance metrics.")

            st.markdown("---")

            # Bands
            if show_bands:
                scenario_fc = add_confidence_bands_logspace(scenario_fc, float(sigma_units_log), z=z)

            st.success(f"✅ Forecast generated (calib factor = {calib:.3f})")

            # Metrics
            st.subheader("📈 Summary Metrics (Scenario)")
            total_units = float(scenario_fc["Predicted_Units"].sum())
            total_revenue = float(scenario_fc["Predicted_Revenue"].sum())
            avg_daily_units = float(scenario_fc["Predicted_Units"].mean())
            roi = total_revenue / float(monthly_spend) if float(monthly_spend) > 0 else 0

            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Total Units", f"{total_units:,.0f}")
            m2.metric("Total Revenue", f"₹{total_revenue/100000:.2f}L")
            m3.metric("Avg Daily Units", f"{avg_daily_units:,.0f}")
            m4.metric("ROI", f"{roi:.2f}x")

            st.subheader("🆚 Scenario Impact vs Baseline")
            delta_units = total_units - float(baseline_fc["Predicted_Units"].sum())
            delta_rev = total_revenue - float(baseline_fc["Predicted_Revenue"].sum())
            c1, c2 = st.columns(2)
            c1.metric("Incremental Units", f"{delta_units:,.0f}")
            c2.metric("Incremental Revenue", f"₹{delta_rev/100000:.2f}L")

            st.markdown("---")

            # History plot (SKU + channel)
            # History plot with improved 30D median
            hist_plot = daily_data[
                (daily_data["Master_Title"] == selected_product) & 
                (daily_data["Channel"] == selected_channel)
            ][["Order_date", "Units_Sold"]].copy().sort_values("Order_date")
            
            # Calculate 30-day median (more stable than 7-day mean)
            hist_plot["Units_30D_Median"] = hist_plot["Units_Sold"].rolling(
                window=30, 
                min_periods=1
            ).median()

            forecast_start = pd.to_datetime(start_date).to_pydatetime()

            fig = plot_history_vs_forecast(
                hist_plot=hist_plot,
                baseline_fc=baseline_fc,
                scenario_fc=scenario_fc,
                forecast_start=forecast_start,
                show_baseline_trend=show_baseline_trend,
                show_bands=show_bands,
            )
            st.plotly_chart(fig, use_container_width=True)

            st.markdown("---")

            st.subheader("📋 Forecast Table (Scenario)")
            display_df = scenario_fc[["Date", "Predicted_Units", "Predicted_Revenue", "Discount_Pct", "Shopify_Discount", "Spend", "Spend_Adstock"]].copy()
            display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
            display_df["Predicted_Units"] = display_df["Predicted_Units"].round(0).astype(int)
            display_df["Predicted_Revenue"] = display_df["Predicted_Revenue"].round(2)
            display_df["Spend_Adstock"] = display_df["Spend_Adstock"].round(2)
            display_df.columns = ["Date", "Units", "Revenue (₹)", "Channel Discount %", "Shopify Discount %", "Spend (₹)", "Spend Adstock"]
            st.dataframe(display_df, use_container_width=True, height=420)

            csv = display_df.to_csv(index=False)
            st.download_button(
                label="📥 Download Forecast CSV (Scenario)",
                data=csv,
                file_name=f"forecast_{selected_product}_{selected_channel}_{start_date}_{end_date}.csv",
                mime="text/csv",
                use_container_width=True,
            )

if __name__ == "__main__":
    main()


