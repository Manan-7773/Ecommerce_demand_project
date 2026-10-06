import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from sklearn.linear_model import LinearRegression

st.set_page_config(page_title="E-Commerce Demand Analysis", page_icon="🛒", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv("data/orders.csv", parse_dates=["date"])
    return df

df = load_data()

st.title(" E-Commerce Product Demand Analysis")
st.caption("PS-04 | Product demand, trend detection, forecasting and inventory insights")

# Sidebar
st.sidebar.header("Filters")
categories = ["All"] + sorted(df["category"].unique().tolist())
cat = st.sidebar.selectbox("Category", categories)
if cat != "All":
    view = df[df["category"] == cat].copy()
else:
    view = df.copy()

min_date, max_date = view["date"].min().date(), view["date"].max().date()
date_range = st.sidebar.date_input("Date range", [min_date, max_date], min_value=min_date, max_value=max_date)
if isinstance(date_range, (list, tuple)) and len(date_range) == 2:
    view = view[(view["date"].dt.date >= date_range[0]) & (view["date"].dt.date <= date_range[1])]

# KPIs
total_units = int(view["units_sold"].sum())
revenue = float(view["revenue"].sum())
products_count = view["product_id"].nunique()
avg_daily = view.groupby("date")["units_sold"].sum().mean()
c1, c2, c3, c4 = st.columns(4)
c1.metric("Units Sold", f"{total_units:,}")
c2.metric("Revenue", f"₹{revenue:,.0f}")
c3.metric("Products", products_count)
c4.metric("Avg. Daily Demand", f"{avg_daily:,.1f}")

st.divider()

# Product performance
st.subheader(" Product Performance")
perf = (view.groupby(["product_id","product_name","category"], as_index=False)
        .agg(units_sold=("units_sold","sum"), revenue=("revenue","sum")))
perf = perf.sort_values("units_sold", ascending=False)

col1, col2 = st.columns(2)
with col1:
    top = perf.head(10)
    fig = px.bar(top, x="units_sold", y="product_name", orientation="h",
                 title="Top Products by Units Sold", text_auto=True)
    fig.update_layout(yaxis={"categoryorder":"total ascending"})
    st.plotly_chart(fig, use_container_width=True)

with col2:
    slow = perf.sort_values("units_sold").head(10)
    fig = px.bar(slow, x="units_sold", y="product_name", orientation="h",
                 title="Slow-Moving Products", text_auto=True)
    fig.update_layout(yaxis={"categoryorder":"total descending"})
    st.plotly_chart(fig, use_container_width=True)

# Trend analysis
st.subheader(" Demand Trend Analysis")
product_options = perf["product_name"].tolist()
selected = st.selectbox("Select product", product_options)
p = view[view["product_name"] == selected].groupby("date", as_index=False)["units_sold"].sum()
p["rolling_7d"] = p["units_sold"].rolling(7).mean()

x = np.arange(len(p)).reshape(-1,1)
y = p["units_sold"].values
model = LinearRegression().fit(x, y)
slope = float(model.coef_[0])
trend_label = "Increasing" if slope > 0.03 else "Decreasing" if slope < -0.03 else "Stable"

fig = px.line(p, x="date", y=["units_sold","rolling_7d"], title=f"{selected} — Daily Demand")
st.plotly_chart(fig, use_container_width=True)
st.info(f"Trend: **{trend_label}** | Estimated daily change: **{slope:+.3f} units/day**")

# Forecast
st.subheader(" Future Demand Forecast")
days = st.slider("Forecast horizon (days)", 7, 60, 30)
recent = p.tail(min(90, len(p))).copy()
x2 = np.arange(len(recent)).reshape(-1,1)
m2 = LinearRegression().fit(x2, recent["units_sold"].values)
future_x = np.arange(len(recent), len(recent)+days).reshape(-1,1)
forecast = np.maximum(0, m2.predict(future_x))
future_dates = pd.date_range(recent["date"].max() + pd.Timedelta(days=1), periods=days)
fc = pd.DataFrame({"date": future_dates, "forecast_units": forecast})

fig = px.line(fc, x="date", y="forecast_units", title=f"{selected} — {days}-Day Forecast")
st.plotly_chart(fig, use_container_width=True)
st.metric("Expected demand over forecast period", f"{fc['forecast_units'].sum():,.0f} units")

# Alerts
st.subheader(" Demand Change Alerts")
alerts = []
for pid, g in view.groupby(["product_id","product_name"]):
    g = g.sort_values("date")
    if len(g) < 30:
        continue
    recent_avg = g.tail(14)["units_sold"].mean()
    previous_avg = g.tail(28).head(14)["units_sold"].mean()
    pct = ((recent_avg - previous_avg) / previous_avg * 100) if previous_avg else 0
    if abs(pct) >= 10:
        alerts.append([pid[0], pid[1], recent_avg, previous_avg, pct,
                       "Increasing" if pct > 0 else "Decreasing"])
alert_df = pd.DataFrame(alerts, columns=["Product ID","Product","Recent Avg","Previous Avg","Change %","Status"])
if alert_df.empty:
    st.success("No significant demand changes detected.")
else:
    st.dataframe(alert_df.sort_values("Change %", key=lambda s: s.abs(), ascending=False),
                 use_container_width=True)

# Inventory recommendations
st.subheader(" Inventory Recommendations")
rec = perf.copy()
rec["avg_daily_demand"] = rec["units_sold"] / max(1, view["date"].nunique())
rec["recommended_stock"] = np.ceil(rec["avg_daily_demand"] * 14).astype(int)
rec["priority"] = np.where(rec["avg_daily_demand"] >= rec["avg_daily_demand"].quantile(.75), "High",
                    np.where(rec["avg_daily_demand"] <= rec["avg_daily_demand"].quantile(.25), "Low", "Medium"))
st.dataframe(rec[["product_name","category","units_sold","avg_daily_demand","recommended_stock","priority"]]
             .sort_values(["priority","units_sold"], ascending=[True,False]), use_container_width=True)

st.caption("Academic/demo project. Forecasts are intentionally simple and should be validated before real inventory decisions.")
