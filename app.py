import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from datetime import datetime
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# =========================================================
# PAGE CONFIG + STYLE
# =========================================================
st.set_page_config(
    page_title="Air Quality Dashboard",
    page_icon="🌍",
    layout="wide"
)

st.markdown("""
<style>
section[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #0f172a, #1e293b);
}
.main { padding-top: 1rem; }
.big-title {
    text-align: center;
    color: #00E5FF;
    font-size: 42px;
    font-weight: 800;
}
.sub-title {
    text-align: center;
    color: #AAAAAA;
    font-size: 18px;
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="big-title">🌍 Air Quality Intelligence Dashboard</div>
<div class="sub-title">Real-Time Monitoring • AI Prediction • Environmental Insights</div>
""", unsafe_allow_html=True)
st.caption("Real-time air pollution monitoring, historical trends, and an educational AQI prediction model.")

# =========================================================
# API KEY
# Set it in .streamlit/secrets.toml:
# OPENWEATHER_API_KEY = "your_new_api_key"
# =========================================================
try:
    API_KEY = st.secrets.get("OPENWEATHER_API_KEY", "")
except Exception:
    API_KEY = ""

# =========================================================
# LOAD HISTORICAL CSV
# =========================================================
@st.cache_data
def load_data():
    data = pd.read_csv("data/air_quality.csv")
    required = ["City", "Date", "AQI"]
    missing = [column for column in required if column not in data.columns]
    if missing:
        raise ValueError(f"CSV is missing required column(s): {', '.join(missing)}")

    data["City"] = data["City"].astype(str).str.strip()
    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")
    data["AQI"] = pd.to_numeric(data["AQI"], errors="coerce")
    if "PM25" in data.columns:
        data["PM25"] = pd.to_numeric(data["PM25"], errors="coerce")
    if "Latitude" in data.columns:
        data["Latitude"] = pd.to_numeric(data["Latitude"], errors="coerce")
    if "Longitude" in data.columns:
        data["Longitude"] = pd.to_numeric(data["Longitude"], errors="coerce")

    data = data.dropna(subset=["City", "Date", "AQI"])
    if data.empty:
        raise ValueError("The CSV has no usable rows after reading City, Date, and AQI.")
    return data.sort_values("Date")

try:
    df = load_data()
except Exception as exc:
    st.error(f"Could not load data/air_quality.csv: {exc}")
    st.info("Check that data/air_quality.csv exists and includes City, Date, and AQI columns.")
    st.stop()

# =========================================================
# SIDEBAR + CITY SELECTION
# =========================================================
st.sidebar.header("⚙️ Dashboard Controls")
cities = sorted(df["City"].dropna().unique().tolist())
if not cities:
    st.error("No cities were found in the CSV.")
    st.stop()

city = st.sidebar.selectbox("Select City", cities)
city_df = df[df["City"] == city].copy().sort_values("Date")
if city_df.empty:
    st.error("No historical records found for the selected city.")
    st.stop()
latest = city_df.iloc[-1]

st.sidebar.markdown("---")
st.sidebar.markdown("## 🌫 Air Quality Monitor")
st.sidebar.markdown("""
Track historical data and live OpenWeather air pollution data.

- Real-time weather
- Live air pollution
- Historical AQI trends
- AI-based trend estimate
""")
st.sidebar.metric("Cities in CSV", df["City"].nunique())
latest_by_city = (
    df.sort_values("Date")
      .groupby("City", as_index=False)
      .tail(1)
      .dropna(subset=["AQI"])
)
if not latest_by_city.empty:
    polluted = latest_by_city.loc[latest_by_city["AQI"].idxmax()]
    cleanest = latest_by_city.loc[latest_by_city["AQI"].idxmin()]
    st.sidebar.error(f"🔴 Highest historical AQI in CSV\n\n{polluted['City']}\n\nAQI: {polluted['AQI']:.0f}")
    st.sidebar.success(f"🟢 Lowest historical AQI in CSV\n\n{cleanest['City']}\n\nAQI: {cleanest['AQI']:.0f}")
st.sidebar.caption(f"Dashboard opened: {datetime.now().strftime('%d-%m-%Y %H:%M')}")
st.sidebar.markdown("---")
st.sidebar.markdown("### Developer\n**Aditya Saw**\n\nTY Data Science\n\nUniversity of Mumbai")

# =========================================================
# FETCH LIVE WEATHER + CURRENT POLLUTION + FORECAST
# =========================================================
weather = None
live_air = None
live_forecast = pd.DataFrame()
weather_error = None
pollution_error = None

if not API_KEY:
    weather_error = (
        "OpenWeather API key not found. Add OPENWEATHER_API_KEY to "
        ".streamlit/secrets.toml, then restart Streamlit."
    )
else:
    try:
        weather_response = requests.get(
            "https://api.openweathermap.org/data/2.5/weather",
            params={"q": city, "appid": API_KEY, "units": "metric"},
            timeout=15
        )
        weather_response.raise_for_status()
        weather = weather_response.json()

        lat = weather["coord"]["lat"]
        lon = weather["coord"]["lon"]

        pollution_response = requests.get(
            "https://api.openweathermap.org/data/2.5/air_pollution",
            params={"lat": lat, "lon": lon, "appid": API_KEY},
            timeout=15
        )
        pollution_response.raise_for_status()
        current_list = pollution_response.json().get("list", [])
        if not current_list:
            raise ValueError("The current air-pollution API returned no records.")

        current = current_list[0]
        current_components = current.get("components", {})
        live_air = {
            "ow_aqi": int(current["main"]["aqi"]),
            "pm25": float(current_components.get("pm2_5", 0)),
            "pm10": float(current_components.get("pm10", 0)),
            "updated": datetime.fromtimestamp(
                int(current.get("dt", datetime.now().timestamp()))
            ).strftime("%d-%m-%Y %H:%M")
        }

        forecast_response = requests.get(
            "https://api.openweathermap.org/data/2.5/air_pollution/forecast",
            params={"lat": lat, "lon": lon, "appid": API_KEY},
            timeout=20
        )
        forecast_response.raise_for_status()
        forecast_items = forecast_response.json().get("list", [])
        forecast_rows = []
        for item in forecast_items:
            parts = item.get("components", {})
            forecast_rows.append({
                "Date & Time": pd.to_datetime(item["dt"], unit="s", utc=True)
                    .tz_convert("Asia/Kolkata").tz_localize(None),
                "OpenWeather AQI (1–5)": item["main"]["aqi"],
                "PM2.5 (µg/m³)": parts.get("pm2_5", np.nan),
                "PM10 (µg/m³)": parts.get("pm10", np.nan)
            })
        live_forecast = pd.DataFrame(forecast_rows)

    except requests.HTTPError as exc:
        response = getattr(exc, "response", None)
        detail = ""
        if response is not None:
            try:
                detail = response.json().get("message", response.text)
            except Exception:
                detail = response.text
        pollution_error = f"OpenWeather HTTP error: {detail or exc}"
    except (requests.RequestException, KeyError, ValueError, TypeError) as exc:
        pollution_error = f"Could not retrieve live data: {exc}"
    except Exception as exc:
        pollution_error = f"Unexpected live-data error: {exc}"

# =========================================================
# REAL-TIME WEATHER
# =========================================================
st.markdown("---")
st.subheader("🌤 Real-Time Weather")
if weather is not None:
    weather_cols = st.columns(4)
    weather_cols[0].metric("🌡 Temperature", f"{weather['main'].get('temp', '—')} °C")
    weather_cols[1].metric("💧 Humidity", f"{weather['main'].get('humidity', '—')}%")
    weather_cols[2].metric("🌬 Wind", f"{weather.get('wind', {}).get('speed', '—')} m/s")
    description = weather.get("weather", [{}])[0].get("description", "Unavailable")
    weather_cols[3].metric("☁️ Condition", description.title())
else:
    st.warning(weather_error or pollution_error or "Live weather is unavailable.")

# =========================================================
# CURRENT LIVE AIR QUALITY
# =========================================================
st.markdown("---")
st.header("🌫 Live Air Pollution Monitoring")
if live_air is not None:
    st.caption(
        f"Live OpenWeather data for {city} • Last API timestamp: {live_air['updated']}. "
        "OpenWeather AQI is on a 1–5 scale; it is not the Indian CPCB AQI scale."
    )
    live_cols = st.columns(4)
    live_cols[0].metric("🌫 OpenWeather AQI", f"{live_air['ow_aqi']} / 5")
    live_cols[1].metric("💨 PM2.5", f"{live_air['pm25']:.2f} µg/m³")
    live_cols[2].metric("🌫 PM10", f"{live_air['pm10']:.2f} µg/m³")
    live_cols[3].metric("🏙 Selected City", city)
else:
    st.error("Current live air-quality data is unavailable.")
    if pollution_error:
        st.caption(pollution_error)
    elif weather_error:
        st.caption(weather_error)

# =========================================================
# LIVE AQI CATEGORY + GAUGE + LIVE STATISTICS
# =========================================================
st.markdown("---")
st.subheader("📈 Live AQI Statistics")
if live_air is not None:
    live_aqi = live_air["ow_aqi"]
    stat_cols = st.columns(4)
    stat_cols[0].metric("Live AQI (1–5)", f"{live_aqi}/5")
    stat_cols[1].metric("Live PM2.5", f"{live_air['pm25']:.2f} µg/m³")
    stat_cols[2].metric("Live PM10", f"{live_air['pm10']:.2f} µg/m³")
    stat_cols[3].metric("Selected City", city)

    categories = {
        1: ("Good", "🟢"),
        2: ("Fair", "🟡"),
        3: ("Moderate", "🟠"),
        4: ("Poor", "🔴"),
        5: ("Very Poor", "🟣")
    }
    category, emoji = categories.get(live_aqi, ("Unknown", "⚪"))
    st.subheader("🌱 Live Air Quality Category")
    if live_aqi <= 2:
        st.success(f"{emoji} {category}")
    elif live_aqi == 3:
        st.info(f"{emoji} {category}")
    else:
        st.warning(f"{emoji} {category}")

    st.subheader("🎯 Live OpenWeather AQI Gauge")
    gauge = go.Figure(go.Indicator(
        mode="gauge+number",
        value=live_aqi,
        number={"suffix": " / 5"},
        title={"text": "OpenWeather AQI"},
        gauge={
            "axis": {"range": [1, 5], "dtick": 1},
            "steps": [
                {"range": [1, 2], "color": "#22c55e"},
                {"range": [2, 3], "color": "#eab308"},
                {"range": [3, 4], "color": "#f97316"},
                {"range": [4, 5], "color": "#ef4444"},
                {"range": [5, 5.01], "color": "#991b1b"}
            ],
            "bar": {"color": "#00E5FF"}
        }
    ))
    gauge.update_layout(height=320, margin={"l": 20, "r": 20, "t": 60, "b": 10})
    st.plotly_chart(gauge, use_container_width=True)
else:
    st.warning("Live statistics and gauge will appear when the live API returns valid data.")

# =========================================================
# HISTORICAL CSV STATISTICS (SEPARATE SCALE/SOURCE)
# =========================================================
st.markdown("---")
st.subheader("📚 Historical AQI Statistics (CSV)")
historical_aqi = pd.to_numeric(city_df["AQI"], errors="coerce").dropna()
if not historical_aqi.empty:
    hist_cols = st.columns(4)
    hist_cols[0].metric("Average Historical AQI", f"{historical_aqi.mean():.2f}")
    hist_cols[1].metric("Maximum Historical AQI", f"{historical_aqi.max():.0f}")
    hist_cols[2].metric("Minimum Historical AQI", f"{historical_aqi.min():.0f}")
    hist_cols[3].metric("CSV Records", len(historical_aqi))
else:
    st.info("No numeric historical AQI values are available for this city.")

# =========================================================
# HISTORICAL AQI + PM2.5 TRENDS
# =========================================================
st.markdown("---")
st.subheader("📉 Historical AQI Trend Analysis (CSV)")
fig_aqi = px.line(
    city_df.dropna(subset=["AQI"]),
    x="Date", y="AQI", markers=True,
    title=f"{city} Historical AQI Trend"
)


if "PM25" in city_df.columns:
    st.subheader("🌫 Historical PM2.5 Trend (CSV)")
    pm_df = city_df.dropna(subset=["PM25"])
    if not pm_df.empty:
        fig_pm = px.line(pm_df, x="Date", y="PM25", markers=True,
                         title=f"{city} Historical PM2.5 Trend")
        st.plotly_chart(fig_pm, use_container_width=True)
    else:
        st.info("No numeric PM25 values found for this city.")
else:
    st.info("The CSV does not contain a PM25 column, so the historical PM2.5 chart is unavailable.")

# =========================================================
# LIVE 4-DAY POLLUTION FORECAST
# =========================================================
st.markdown("---")
st.subheader("🔮 Next 4 Days — Live Pollution Forecast")
if not live_forecast.empty:
    chart_cols = [c for c in ["PM2.5 (µg/m³)", "PM10 (µg/m³)"] if c in live_forecast.columns]
    if chart_cols:
        st.line_chart(live_forecast.set_index("Date & Time")[chart_cols])
    st.dataframe(live_forecast, use_container_width=True)
    st.download_button(
        "⬇️ Download Live Pollution Forecast",
        data=live_forecast.to_csv(index=False).encode("utf-8"),
        file_name=f"{city.replace(' ', '_')}_pollution_forecast.csv",
        mime="text/csv"
    )
else:
    st.info("Live pollution forecast is unavailable until the API request succeeds.")

# =========================================================
# CITY VS CITY COMPARISON (HISTORICAL CSV)
# =========================================================
st.markdown("---")
st.subheader("🏙️ City vs City Comparison (Historical CSV)")
comparison_cities = sorted(df["City"].dropna().unique().tolist())
if len(comparison_cities) >= 2:
    default_second = 1
    city_1 = st.selectbox("Select First City", comparison_cities, key="city1")
    city_2 = st.selectbox("Select Second City", comparison_cities, index=default_second, key="city2")
    compare_df = df[df["City"].isin([city_1, city_2])]
    fig_compare = px.line(
        compare_df, x="Date", y="AQI", color="City", markers=True,
        title=f"{city_1} vs {city_2} Historical AQI"
    )
    st.plotly_chart(fig_compare, use_container_width=True)
else:
    st.info("At least two cities are needed for city comparison.")

# =========================================================
# AI AQI TREND ESTIMATE FROM HISTORICAL CSV
# =========================================================
st.markdown("---")
st.subheader("🤖 AI-Based AQI Trend Estimate (Historical CSV)")
model_df = city_df.dropna(subset=["AQI", "Date"]).sort_values("Date").copy()
if len(model_df) >= 2:
    model_df["Day"] = np.arange(len(model_df))
    X = model_df[["Day"]]
    y = model_df["AQI"].astype(float)
    model = LinearRegression()
    model.fit(X, y)
    training_predictions = model.predict(X)
    accuracy = r2_score(y, training_predictions) if len(y) >= 2 else np.nan

    model_cols = st.columns(2)
    model_cols[0].metric("Training R² Score", f"{accuracy:.2f}" if np.isfinite(accuracy) else "N/A")
    tomorrow_input = pd.DataFrame({"Day": [len(model_df)]})
    predicted_aqi = float(model.predict(tomorrow_input)[0])
    model_cols[1].metric("Tomorrow's Estimated Historical-Scale AQI", f"{predicted_aqi:.2f}")

    if predicted_aqi <= 50:
        st.success("Estimated category using the app's historical thresholds: Good")
    elif predicted_aqi <= 100:
        st.info("Estimated category using the app's historical thresholds: Moderate")
    elif predicted_aqi <= 200:
        st.warning("Estimated category using the app's historical thresholds: Poor")
    else:
        st.error("Estimated category using the app's historical thresholds: Very Poor")

    future_days = pd.DataFrame({"Day": np.arange(len(model_df), len(model_df) + 7)})
    future_predictions = model.predict(future_days)
    forecast_df = pd.DataFrame({
        "Day": [f"Day {i}" for i in range(1, 8)],
        "Predicted AQI (historical scale)": future_predictions
    })
    fig_future = px.line(
        forecast_df, x="Day", y="Predicted AQI (historical scale)",
        markers=True, title="7-Day Historical-Scale AQI Trend Estimate"
    )
    st.plotly_chart(fig_future, use_container_width=True)
    st.dataframe(forecast_df, use_container_width=True)
else:
    st.info("At least two valid historical AQI records are required to train the trend model.")
    predicted_aqi = None
    accuracy = np.nan

# =========================================================
# HISTORICAL CITY INSIGHTS
# =========================================================
st.markdown("---")
st.subheader("🌎 Historical Environmental Insights")
if not latest_by_city.empty:
    most_polluted = latest_by_city.loc[latest_by_city["AQI"].idxmax()]
    cleanest_city = latest_by_city.loc[latest_by_city["AQI"].idxmin()]
    average_aqi = latest_by_city["AQI"].mean()
    st.info(
        f"🔴 Highest latest historical AQI in CSV: {most_polluted['City']} "
        f"(AQI {most_polluted['AQI']:.0f})\n\n"
        f"🟢 Lowest latest historical AQI in CSV: {cleanest_city['City']} "
        f"(AQI {cleanest_city['AQI']:.0f})\n\n"
        f"Average of latest city records: {average_aqi:.2f}\n\n"
        "These insights are calculated from the historical CSV, not from live API measurements."
    )

# =========================================================
# INDIA POLLUTION MAP (HISTORICAL CSV)
# =========================================================
st.markdown("---")
st.subheader("🗺️ India Air Quality Intelligence Map (Historical CSV)")
required_map_cols = {"City", "Latitude", "Longitude", "AQI", "Date"}
if required_map_cols.issubset(df.columns):
    map_df = (
        df.dropna(subset=["Latitude", "Longitude", "AQI", "Date"])
          .sort_values("Date")
          .groupby("City", as_index=False)
          .tail(1)
          .copy()
    )
    if not map_df.empty:
        map_df["AQI Category"] = pd.cut(
            map_df["AQI"],
            bins=[-np.inf, 50, 100, 200, 300, np.inf],
            labels=["Good", "Moderate", "Poor", "Very Poor", "Severe"]
        ).astype(str)
        color_map = {
            "Good": "#22c55e", "Moderate": "#eab308",
            "Poor": "#f97316", "Very Poor": "#ef4444", "Severe": "#991b1b"
        }
        
        fig_map = go.Figure()

        for category in map_df["AQI Category"].dropna().unique():
            category_df = map_df[
                map_df["AQI Category"] == category
            ]

            fig_map.add_trace(
                go.Scattergeo(
                    lat=category_df["Latitude"],
                    lon=category_df["Longitude"],
                    text=category_df["City"],
                    mode="markers",
                    name=str(category),
                    marker={
                        "size": [
                            max(8, min(24, float(aqi) / 8))
                            for aqi in category_df["AQI"]
                        ],
                        "color": color_map.get(category, "#64748b"),
                        "opacity": 0.85,
                        "line": {"width": 1, "color": "white"}
                    },
                    customdata=category_df[["AQI"]].to_numpy(),
                    hovertemplate=(
                        "<b>%{text}</b><br>"
                        "AQI: %{customdata[0]}<br>"
                        "Category: " + str(category) +
                        "<extra></extra>"
                    )
                )
            )

        fig_map.update_geos(
            scope="asia",
            center={"lat": 22.5, "lon": 79.0},
            projection_scale=3.5,
            showland=True,
            landcolor="#1e293b",
            showcountries=True,
            countrycolor="#64748b",
            showocean=True,
            oceancolor="#0f172a",
            showcoastlines=True
        )

        fig_map.update_layout(
            height=550,
            title="Latest Historical AQI Across Indian Cities",
            margin={"r": 0, "t": 50, "l": 0, "b": 0},
            legend_title_text="Pollution Level"
        )

        st.plotly_chart(
            fig_map,
            use_container_width=True
        )

        st.plotly_chart(fig_map, use_container_width=True, key="historical_aqi_map")
    else:
        st.info("No usable latitude/longitude values are available in the CSV.")
else:
    st.info("Map requires City, Latitude, Longitude, AQI, and Date columns in the CSV.")

# =========================================================
# CITY RANKING + CHARTS
# =========================================================
st.markdown("---")
st.subheader("🏆 City AQI Ranking (Historical CSV)")
if not latest_by_city.empty:
    ranking_df = latest_by_city[["City", "AQI"]].sort_values("AQI", ascending=False).reset_index(drop=True)
    ranking_df.insert(0, "Rank", np.arange(1, len(ranking_df) + 1))
    st.dataframe(ranking_df, use_container_width=True)

    fig_rank = px.bar(ranking_df, x="City", y="AQI", color="AQI", title="Latest Historical AQI by City")
    st.plotly_chart(fig_rank, use_container_width=True)

    st.subheader("🔴 Top 5 Most Polluted Cities (Historical CSV)")
    st.plotly_chart(
        px.bar(ranking_df.head(5), x="City", y="AQI", color="AQI", title="Top 5 Highest Historical AQI"),
        use_container_width=True
    )

    st.subheader("🟢 Top 5 Cleanest Cities (Historical CSV)")
    clean5 = ranking_df.sort_values("AQI", ascending=True).head(5)
    st.plotly_chart(
        px.bar(clean5, x="City", y="AQI", color="AQI", title="Top 5 Lowest Historical AQI"),
        use_container_width=True
    )

# =========================================================
# POLLUTION ALERT (BASED ON HISTORICAL MODEL ONLY)
# =========================================================
st.markdown("---")
st.subheader("🚨 Pollution Alert System")
if predicted_aqi is None:
    st.info("The alert needs at least two historical AQI records for the selected city.")
elif predicted_aqi > 200:
    st.error("HIGH POLLUTION ALERT — Avoid prolonged outdoor activities, especially if sensitive to pollution.")
elif predicted_aqi > 100:
    st.warning("POLLUTION ALERT — Sensitive people should consider reducing prolonged outdoor exposure.")
else:
    st.success("The model's historical-scale estimate is at or below 100. This is not a real-time health advisory.")

# =========================================================
# DATASET PREVIEW
# =========================================================
st.markdown("---")
st.subheader("🧾 Dataset Preview")
with st.expander("View Dataset Information"):
    info_cols = st.columns(3)
    info_cols[0].metric("Total Records", len(df))
    info_cols[1].metric("Total Cities", df["City"].nunique())
    info_cols[2].metric("Unique Dates", df["Date"].nunique())
    st.markdown("### Dataset Columns")
    st.write(list(df.columns))
    st.markdown(f"### {city} — Recent Historical AQI Data")
    st.dataframe(city_df.tail(20), use_container_width=True)

# =========================================================
# PROJECT SUMMARY + DOWNLOAD
# =========================================================
st.markdown("---")
st.subheader("📋 Project Summary")
summary = {
    "Selected City": city,
    "Latest Historical CSV AQI": latest["AQI"],
    "Live OpenWeather AQI (1–5)": live_air["ow_aqi"] if live_air is not None else "Unavailable",
    "Live PM2.5 (µg/m³)": live_air["pm25"] if live_air is not None else "Unavailable",
    "Live PM10 (µg/m³)": live_air["pm10"] if live_air is not None else "Unavailable",
    "Tomorrow Historical-Scale Estimate": round(predicted_aqi, 2) if predicted_aqi is not None else "Unavailable",
    "Model Training R²": round(accuracy, 2) if np.isfinite(accuracy) else "Unavailable"
}
st.dataframe(pd.DataFrame([summary]), use_container_width=True)

st.subheader("⬇️ Download Historical AQI Report")
st.download_button(
    label="Download Selected City's Historical CSV Report",
    data=city_df.to_csv(index=False).encode("utf-8"),
    file_name=f"{city.replace(' ', '_')}_historical_AQI_report.csv",
    mime="text/csv"
)

st.markdown("""
---
### 🌍 Air Quality Trend Analysis System
Developed using Python, Streamlit, Machine Learning, Plotly, and OpenWeather API.

**TY Data Science Final Year Project**
""")
