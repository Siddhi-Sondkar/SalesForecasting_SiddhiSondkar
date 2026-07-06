"""
Superstore Sales Forecasting — Streamlit Dashboard
Run locally with:  streamlit run app.py
Deploy free on Streamlit Community Cloud by pointing it at this file in your GitHub repo.
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Superstore Sales Forecasting", layout="wide")

# --------------------------------------------------------------------------------------
# Data loading & caching
# --------------------------------------------------------------------------------------

@st.cache_data
def load_data():
    df = pd.read_csv("train.csv")
    df["Order Date"] = pd.to_datetime(df["Order Date"], dayfirst=True)
    df["Ship Date"] = pd.to_datetime(df["Ship Date"], dayfirst=True)
    df["Order Year"] = df["Order Date"].dt.year
    df["Order Month"] = df["Order Date"].dt.month
    df["Order Quarter"] = df["Order Date"].dt.quarter

    def season(m):
        if m in [12, 1, 2]:
            return "Winter"
        if m in [3, 4, 5]:
            return "Spring"
        if m in [6, 7, 8]:
            return "Summer"
        return "Fall"

    df["Season"] = df["Order Month"].apply(season)
    return df


def season_num(m):
    if m in [12, 1, 2]:
        return 0
    if m in [3, 4, 5]:
        return 1
    if m in [6, 7, 8]:
        return 2
    return 3


@st.cache_data
def get_monthly_series(df, col=None, val=None):
    d = df if col is None else df[df[col] == val]
    return d.set_index("Order Date").resample("MS")["Sales"].sum()


@st.cache_data
def get_weekly_series(df):
    return df.set_index("Order Date").resample("W")["Sales"].sum()


# --------------------------------------------------------------------------------------
# Forecasting models (mirrors analysis.ipynb Task 3 / Task 4 logic)
# --------------------------------------------------------------------------------------

FEATURES = ["lag1", "lag2", "lag3", "roll3", "month", "quarter", "season"]


def forecast_sarima(series, steps=3):
    from statsmodels.tsa.statespace.sarimax import SARIMAX
    fit = SARIMAX(series, order=(1, 1, 1), seasonal_order=(1, 1, 1, 12),
                  enforce_stationarity=False, enforce_invertibility=False).fit(disp=False)
    fc = fit.get_forecast(steps=steps)
    return fc.predicted_mean.values, fc.conf_int().values


def forecast_prophet(series, steps=3):
    from prophet import Prophet
    pdf = series.reset_index()
    pdf.columns = ["ds", "y"]
    m = Prophet(yearly_seasonality=True, weekly_seasonality=False, daily_seasonality=False)
    m.fit(pdf)
    future = m.make_future_dataframe(periods=steps, freq="MS")
    fc = m.predict(future)
    tail = fc.iloc[-steps:]
    return tail["yhat"].values, tail[["yhat_lower", "yhat_upper"]].values


def forecast_xgboost(series, steps=3):
    from xgboost import XGBRegressor
    d = series.reset_index()
    d.columns = ["ds", "y"]
    d["month"] = d["ds"].dt.month
    d["quarter"] = d["ds"].dt.quarter
    d["season"] = d["month"].apply(season_num)
    d["lag1"] = d["y"].shift(1)
    d["lag2"] = d["y"].shift(2)
    d["lag3"] = d["y"].shift(3)
    d["roll3"] = d["y"].shift(1).rolling(3).mean()
    d = d.dropna().reset_index(drop=True)
    model = XGBRegressor(n_estimators=200, max_depth=3, learning_rate=0.05, random_state=42)
    model.fit(d[FEATURES], d["y"])
    history = d["y"].tolist()
    last_date = d["ds"].iloc[-1]
    preds = []
    for _ in range(steps):
        next_date = last_date + pd.DateOffset(months=1)
        row = pd.DataFrame([[history[-1], history[-2], history[-3], np.mean(history[-3:]),
                              next_date.month, next_date.quarter, season_num(next_date.month)]],
                            columns=FEATURES)
        p = model.predict(row)[0]
        preds.append(p)
        history.append(p)
        last_date = next_date
    return np.array(preds), None


MODEL_FUNCS = {"SARIMA": forecast_sarima, "Prophet": forecast_prophet, "XGBoost": forecast_xgboost}


def eval_metrics(actual, pred):
    actual, pred = np.array(actual), np.array(pred)
    mae = np.mean(np.abs(pred - actual))
    rmse = np.sqrt(np.mean((pred - actual) ** 2))
    mape = np.mean(np.abs((pred - actual) / actual)) * 100
    return mae, rmse, mape


@st.cache_data
def evaluate_model(model_name, _series_key, series):
    """Holdout-evaluate a model on the last 3 months of `series`."""
    train, test = series.iloc[:-3], series.iloc[-3:]
    fn = MODEL_FUNCS[model_name]
    pred, _ = fn(train, steps=3)
    return eval_metrics(test.values, pred)


# --------------------------------------------------------------------------------------
# Sidebar navigation
# --------------------------------------------------------------------------------------

st.sidebar.title("📊 Superstore Analytics")
page = st.sidebar.radio(
    "Go to",
    ["Sales Overview", "Forecast Explorer", "Anomaly Report", "Product Demand Segments"],
)

df = load_data()

# --------------------------------------------------------------------------------------
# PAGE 1 — Sales Overview Dashboard
# --------------------------------------------------------------------------------------

if page == "Sales Overview":
    st.title("Sales Overview Dashboard")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Total Sales by Year")
        yearly = df.groupby("Order Year")["Sales"].sum()
        fig, ax = plt.subplots()
        yearly.plot(kind="bar", ax=ax, color="#4C72B0")
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    with col2:
        st.subheader("Monthly Sales Trend")
        monthly = get_monthly_series(df)
        fig, ax = plt.subplots()
        monthly.plot(ax=ax, color="#DD8452")
        ax.set_ylabel("Sales ($)")
        st.pyplot(fig)

    st.subheader("Sales by Region & Category")
    c1, c2 = st.columns(2)
    with c1:
        regions = st.multiselect("Filter by Region", sorted(df["Region"].unique()),
                                  default=sorted(df["Region"].unique()))
    with c2:
        categories = st.multiselect("Filter by Category", sorted(df["Category"].unique()),
                                     default=sorted(df["Category"].unique()))

    filtered = df[df["Region"].isin(regions) & df["Category"].isin(categories)]
    pivot = filtered.groupby(["Region", "Category"])["Sales"].sum().unstack(fill_value=0)
    st.bar_chart(pivot)
    st.dataframe(pivot.style.format("${:,.0f}"))


# --------------------------------------------------------------------------------------
# PAGE 2 — Forecast Explorer
# --------------------------------------------------------------------------------------

elif page == "Forecast Explorer":
    st.title("Forecast Explorer")

    c1, c2, c3 = st.columns(3)
    with c1:
        dim = st.selectbox("Select dimension", ["Category", "Region"])
    with c2:
        value = st.selectbox(f"Select {dim}", sorted(df[dim].unique()))
    with c3:
        horizon = st.select_slider("Forecast horizon (months ahead)", options=[1, 2, 3], value=3)

    model_name = st.selectbox("Model", ["SARIMA", "Prophet", "XGBoost"], index=0)

    series = get_monthly_series(df, dim, value)

    with st.spinner(f"Fitting {model_name} on {value}..."):
        mae, rmse, mape = evaluate_model(model_name, f"{dim}-{value}", series)
        forecast, ci = MODEL_FUNCS[model_name](series, steps=3)

    forecast_dates = pd.date_range(series.index[-1] + pd.DateOffset(months=1), periods=3, freq="MS")

    fig, ax = plt.subplots(figsize=(10, 5))
    series.plot(ax=ax, label="Actual")
    ax.plot(forecast_dates[:horizon], forecast[:horizon], marker="o", linestyle="--",
            color="#C44E52", label=f"{model_name} Forecast")
    if ci is not None:
        ax.fill_between(forecast_dates[:horizon], ci[:horizon, 0], ci[:horizon, 1],
                         color="#C44E52", alpha=0.2, label="Confidence interval")
    ax.set_title(f"{value} — {model_name} Forecast ({horizon}-month horizon)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Forecast values")
    st.dataframe(pd.DataFrame({"Date": forecast_dates[:horizon].date,
                                "Forecast ($)": np.round(forecast[:horizon], 0)}))

    st.subheader("Model accuracy (holdout: last 3 actual months)")
    m1, m2, m3 = st.columns(3)
    m1.metric("MAE", f"${mae:,.0f}")
    m2.metric("RMSE", f"${rmse:,.0f}")
    m3.metric("MAPE", f"{mape:.1f}%")


# --------------------------------------------------------------------------------------
# PAGE 3 — Anomaly Report
# --------------------------------------------------------------------------------------

elif page == "Anomaly Report":
    st.title("Anomaly Report")

    from sklearn.ensemble import IsolationForest

    weekly = get_weekly_series(df).reset_index()
    weekly.columns = ["ds", "y"]

    iso = IsolationForest(contamination=0.05, random_state=42)
    weekly["anomaly_iso"] = iso.fit_predict(weekly[["y"]])

    roll_mean = weekly["y"].rolling(8, min_periods=4).mean()
    roll_std = weekly["y"].rolling(8, min_periods=4).std()
    weekly["zscore"] = (weekly["y"] - roll_mean) / roll_std
    weekly["anomaly_zscore"] = weekly["zscore"].abs() > 2

    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(weekly["ds"], weekly["y"], label="Weekly Sales", color="#4C72B0")
    iso_pts = weekly[weekly["anomaly_iso"] == -1]
    z_pts = weekly[weekly["anomaly_zscore"]]
    ax.scatter(iso_pts["ds"], iso_pts["y"], color="red", marker="X", s=90, label="Isolation Forest anomaly", zorder=5)
    ax.scatter(z_pts["ds"], z_pts["y"], color="orange", marker="D", s=70, label="Z-score anomaly", zorder=4)
    ax.set_title("Weekly Sales — Detected Anomalies")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Isolation Forest anomalies")
    st.dataframe(iso_pts[["ds", "y"]].rename(columns={"ds": "Week", "y": "Sales ($)"}))

    st.subheader("Z-score anomalies")
    st.dataframe(z_pts[["ds", "y", "zscore"]].rename(columns={"ds": "Week", "y": "Sales ($)", "zscore": "Z-score"}))


# --------------------------------------------------------------------------------------
# PAGE 4 — Product Demand Segments
# --------------------------------------------------------------------------------------

elif page == "Product Demand Segments":
    st.title("Product Demand Segments")

    from sklearn.preprocessing import StandardScaler
    from sklearn.cluster import KMeans
    from sklearn.decomposition import PCA

    sub_agg = df.groupby("Sub-Category").agg(total_sales=("Sales", "sum")).reset_index()

    monthly_sc = df.groupby(["Sub-Category", df["Order Date"].dt.to_period("M")])["Sales"].sum().reset_index()
    monthly_sc.columns = ["Sub-Category", "Month", "Sales"]
    volatility = monthly_sc.groupby("Sub-Category")["Sales"].std().reset_index().rename(columns={"Sales": "volatility"})

    yearly_sc = df.groupby(["Sub-Category", "Order Year"])["Sales"].sum().reset_index()

    def yoy_growth(g):
        g = g.sort_values("Order Year")
        if len(g) < 2 or g["Sales"].iloc[0] == 0:
            return 0.0
        return (g["Sales"].iloc[-1] - g["Sales"].iloc[0]) / g["Sales"].iloc[0]

    growth = yearly_sc.groupby("Sub-Category").apply(yoy_growth, include_groups=False).reset_index()
    growth.columns = ["Sub-Category", "growth_rate"]

    order_val = df.groupby("Sub-Category").agg(avg_order_value=("Sales", "mean")).reset_index()

    feat = sub_agg.merge(volatility, on="Sub-Category").merge(growth, on="Sub-Category").merge(order_val, on="Sub-Category")
    feature_cols = ["total_sales", "growth_rate", "volatility", "avg_order_value"]
    X_scaled = StandardScaler().fit_transform(feat[feature_cols])

    k = st.slider("Number of clusters (k)", 2, 7, 4)
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
    feat["cluster"] = kmeans.fit_predict(X_scaled)

    profile = feat.groupby("cluster")[feature_cols].mean()

    def label_cluster(row, profile):
        high_vol = row["total_sales"] >= profile["total_sales"].median()
        high_volat = row["volatility"] >= profile["volatility"].median()
        high_growth = row["growth_rate"] >= profile["growth_rate"].median()
        if high_vol and not high_volat:
            return "High Volume, Stable Demand"
        if not high_vol and high_volat:
            return "Low Volume, High Volatility"
        if high_growth:
            return "Growing Demand"
        return "Declining Demand"

    labels = {c: label_cluster(profile.loc[c], profile) for c in profile.index}
    feat["cluster_label"] = feat["cluster"].map(labels)

    pca = PCA(n_components=2)
    pcs = pca.fit_transform(X_scaled)
    feat["pc1"], feat["pc2"] = pcs[:, 0], pcs[:, 1]

    fig, ax = plt.subplots(figsize=(9, 6))
    for label in feat["cluster_label"].unique():
        sub = feat[feat["cluster_label"] == label]
        ax.scatter(sub["pc1"], sub["pc2"], s=100, label=label)
    for _, row in feat.iterrows():
        ax.annotate(row["Sub-Category"], (row["pc1"], row["pc2"]), fontsize=8, xytext=(4, 4), textcoords="offset points")
    ax.set_title("Product Sub-Category Demand Clusters (PCA)")
    ax.legend()
    st.pyplot(fig)

    st.subheader("Sub-categories by cluster")
    st.dataframe(
        feat[["Sub-Category", "cluster_label", "total_sales", "growth_rate", "volatility", "avg_order_value"]]
        .sort_values("cluster_label")
        .style.format({"total_sales": "${:,.0f}", "growth_rate": "{:.1%}",
                        "volatility": "${:,.0f}", "avg_order_value": "${:,.0f}"})
    )
