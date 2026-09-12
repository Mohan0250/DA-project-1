import sys
from pathlib import Path
import streamlit as st
import pandas as pd
from recommendation_engine import generate_recommendations
import folium
import numpy as np
import plotly.express as px
from streamlit_folium import st_folium

# Allow imports from the dashboard folder.
sys.path.append(str(Path(__file__).resolve().parent))

from live_data_pipeline import get_live_city_data
from predictive_engine import (
    get_feature_importance,
    get_prediction_vs_actual,
    predict_next_stress,
)


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Coimbatore City Intelligence",
    page_icon="🏙️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# PROJECT CONSTANTS
# ============================================================

CITY_NAME = "Coimbatore"
LATITUDE = 11.0168
LONGITUDE = 76.9558

HISTORICAL_FILE = r"C:\Users\mohan\dashboard_data.csv"

PM25_MIN = 5
PM25_MAX = 25

TEMP_MIN = 20
TEMP_MAX = 40

HUMIDITY_MIN = 20
HUMIDITY_MAX = 90


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def normalize_score(value, minimum, maximum):
    """Min-max normalization clipped to 0-1."""
    if maximum == minimum:
        return 0.0

    score = (float(value) - minimum) / (maximum - minimum)
    return float(np.clip(score, 0, 1))


def calculate_live_stress(data):
    """Calculate project-defined City Stress components."""
    pollution_score = normalize_score(
        data["pm2_5"],
        PM25_MIN,
        PM25_MAX,
    )

    heat_score = normalize_score(
        data["temperature"],
        TEMP_MIN,
        TEMP_MAX,
    )

    humidity_score = normalize_score(
        data["humidity"],
        HUMIDITY_MIN,
        HUMIDITY_MAX,
    )

    weather_stress_score = (
        0.7 * heat_score
        + 0.3 * humidity_score
    )

    congestion_ratio = float(
        np.clip(data.get("congestion_ratio", 0), 0, 1)
    )

    city_stress_index = (
        0.40 * pollution_score
        + 0.35 * congestion_ratio
        + 0.25 * weather_stress_score
    )

    city_stress_score = city_stress_index * 100

    if city_stress_score < 25:
        stress_level = "Low"
    elif city_stress_score < 50:
        stress_level = "Moderate"
    elif city_stress_score < 75:
        stress_level = "High"
    else:
        stress_level = "Critical"

    return {
        "pollution_score": pollution_score,
        "heat_score": heat_score,
        "humidity_score": humidity_score,
        "weather_stress_score": weather_stress_score,
        "city_stress_index": city_stress_index,
        "city_stress_score": city_stress_score,
        "stress_level": stress_level,
    }


def stress_color(stress_level):
    """Folium marker color."""
    return {
        "Low": "green",
        "Moderate": "orange",
        "High": "red",
        "Critical": "darkred",
    }.get(stress_level, "blue")


def load_historical_data():
    """Load and prepare historical dashboard data."""
    path = Path(HISTORICAL_FILE)

    if not path.exists():
        return pd.DataFrame()

    df = pd.read_csv(path)

    if "timestamp" not in df.columns:
        return pd.DataFrame()

    df["timestamp"] = pd.to_datetime(
        df["timestamp"],
        errors="coerce",
    )

    df = df.dropna(subset=["timestamp"]).sort_values("timestamp")

    numeric_columns = [
        "temperature",
        "humidity",
        "wind_speed",
        "pm2_5",
        "pm10",
        "us_aqi",
        "current_speed",
        "free_flow_speed",
        "congestion_ratio",
        "city_stress_score",
    ]

    for column in numeric_columns:
        if column in df.columns:
            df[column] = pd.to_numeric(
                df[column],
                errors="coerce",
            )

    if "congestion_ratio" not in df.columns:
        if {
            "current_speed",
            "free_flow_speed",
        }.issubset(df.columns):
            df["congestion_ratio"] = (
                1
                - (
                    df["current_speed"]
                    / df["free_flow_speed"].replace(0, np.nan)
                )
            ).clip(0, 1)

    return df.drop_duplicates(
        subset=["timestamp"],
        keep="last",
    ).reset_index(drop=True)


def add_time_gap_breaks(df, timestamp_col="timestamp", max_gap_minutes=30):
    """
    Insert NaN rows between observations separated by more than the allowed gap.
    This prevents historical charts from implying data existed during missing periods.
    """
    if df is None or df.empty or timestamp_col not in df.columns:
        return df

    result = df.copy()
    result[timestamp_col] = pd.to_datetime(result[timestamp_col], errors="coerce")
    result = result.dropna(subset=[timestamp_col]).sort_values(timestamp_col)

    if len(result) < 2:
        return result

    rows = []
    previous_time = None

    for _, row in result.iterrows():
        current_time = row[timestamp_col]

        if (
            previous_time is not None
            and current_time - previous_time
            > pd.Timedelta(minutes=max_gap_minutes)
        ):
            gap_row = {column: np.nan for column in result.columns}
            gap_row[timestamp_col] = previous_time + (
                current_time - previous_time
            ) / 2
            rows.append(gap_row)

        rows.append(row.to_dict())
        previous_time = current_time

    return pd.DataFrame(rows)


def make_chart_df(df, columns, rename=None):
    """Return clean timestamp-indexed chart data."""
    available = [
        column
        for column in columns
        if column in df.columns
    ]

    chart_df = df[
        ["timestamp"] + available
    ].dropna(
        subset=available,
        how="all",
    ).copy()

    if chart_df.empty:
        return chart_df

    chart_df = chart_df.set_index("timestamp")

    if rename:
        chart_df = chart_df.rename(columns=rename)

    return chart_df


# ============================================================
# HEADER
# ============================================================

st.title("🏙️ Coimbatore City Intelligence Dashboard")

st.markdown(
    "Real-time weather, air quality, traffic, and City Stress "
    "monitoring with predictive analytics."
)

st.caption(
    "📡 Live APIs • 🤖 Machine Learning • 🗺️ Spatial Intelligence "
    "• 📊 Historical Analytics"
)


# ============================================================
# LOAD HISTORICAL DATA FIRST
# ============================================================

historical_data = load_historical_data()
historical_data_available = not historical_data.empty


# ============================================================
# LIVE DATA
# ============================================================

live_data = None
live_error = None

try:
    live_data = get_live_city_data()
except Exception as exc:
    live_error = str(exc)


# ============================================================
# REFRESH CONTROL
# ============================================================

if st.button("🔄 Refresh Live Data"):
    st.rerun()


# ============================================================
# FALLBACK LOGIC
# ============================================================

using_historical_fallback = False

if live_data is None and historical_data_available:
    latest_row = historical_data.iloc[-1]

    fallback_data = {
        "timestamp": latest_row["timestamp"].strftime(
            "%Y-%m-%dT%H:%M"
        ),
        "temperature": latest_row.get("temperature", np.nan),
        "humidity": latest_row.get("humidity", np.nan),
        "wind_speed": latest_row.get("wind_speed", np.nan),
        "pm2_5": latest_row.get("pm2_5", np.nan),
        "pm10": latest_row.get("pm10", np.nan),
        "us_aqi": latest_row.get("us_aqi", np.nan),
        "current_speed": latest_row.get("current_speed", 0),
        "free_flow_speed": latest_row.get(
            "free_flow_speed",
            latest_row.get("current_speed", 0),
        ),
        "congestion_ratio": latest_row.get(
            "congestion_ratio",
            0,
        ),
    }

    live_data = fallback_data
    using_historical_fallback = True


# ============================================================
# LIVE STATUS
# ============================================================

if live_error and using_historical_fallback:
    st.warning(
        "⚠️ Live API temporarily unavailable. "
        "Showing the latest historical data so the dashboard "
        "remains available."
    )
elif live_error and not historical_data_available:
    st.error("❌ Unable to retrieve live data.")
    st.error(f"Error: {live_error}")
else:
    st.success("🟢 Live API data connected")


# ============================================================
# LIVE CALCULATIONS
# ============================================================

stress = None

if live_data is not None:
    stress = calculate_live_stress(live_data)

    city_stress_score = stress["city_stress_score"]
    stress_level = stress["stress_level"]

    temperature = float(live_data["temperature"])
    humidity = float(live_data["humidity"])
    pm2_5 = float(live_data["pm2_5"])
    pm10 = float(live_data.get("pm10", 0))
    us_aqi = float(live_data.get("us_aqi", 0))
    wind_speed = float(live_data.get("wind_speed", 0))
    current_speed = float(live_data.get("current_speed", 0))
    free_flow_speed = float(
        live_data.get(
            "free_flow_speed",
            current_speed,
        )
    )
    congestion_ratio = float(
        live_data.get("congestion_ratio", 0)
    )

    pollution_score = stress["pollution_score"] * 100
    traffic_score = congestion_ratio * 100
    weather_score = stress["weather_stress_score"] * 100


# ============================================================
# CURRENT CONDITIONS
# ============================================================

if live_data is not None:

    st.subheader("📍 Latest City Conditions")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "🌡️ Temperature",
            f"{temperature:.1f} °C",
        )

    with col2:
        st.metric(
            "💧 Humidity",
            f"{humidity:.0f} %",
        )

    with col3:
        st.metric(
            "🌫️ PM2.5",
            f"{pm2_5:.1f}",
        )

    with col4:
        st.metric(
            "🏙️ City Stress",
            f"{city_stress_score:.1f}",
        )


    # ========================================================
    # CURRENT STRESS LEVEL
    # ========================================================

    st.subheader("Current Stress Level")

    stress_messages = {
        "Low": "🟢 Low",
        "Moderate": "🟡 Moderate",
        "High": "🟠 High",
        "Critical": "🔴 Critical",
    }

    if stress_level == "Low":
        st.success(stress_messages[stress_level])
    elif stress_level == "Moderate":
        st.info(stress_messages[stress_level])
    elif stress_level == "High":
        st.warning(stress_messages[stress_level])
    else:
        st.error(stress_messages[stress_level])


    # ========================================================
    # INTELLIGENT CITY ALERTS
    # ========================================================

    st.subheader("🚨 Intelligent City Alerts")

    alerts = []

    if city_stress_score >= 75:
        alerts.append(
            (
                "critical",
                "🔴 CRITICAL: Overall City Stress is very high.",
            )
        )
    elif city_stress_score >= 50:
        alerts.append(
            (
                "warning",
                "🟠 WARNING: Overall City Stress is elevated.",
            )
        )

    if pollution_score >= 75:
        alerts.append(
            (
                "critical",
                "🌫️ HIGH POLLUTION: Air pollution is strongly "
                "contributing to City Stress.",
            )
        )
    elif pollution_score >= 50:
        alerts.append(
            (
                "warning",
                "🌫️ MODERATE POLLUTION: Air quality requires attention.",
            )
        )

    if traffic_score >= 75:
        alerts.append(
            (
                "critical",
                "🚗 SEVERE CONGESTION: Traffic congestion is very high.",
            )
        )
    elif traffic_score >= 50:
        alerts.append(
            (
                "warning",
                "🚗 HIGH CONGESTION: Traffic conditions may affect "
                "City Stress.",
            )
        )

    if weather_score >= 75:
        alerts.append(
            (
                "critical",
                "🌡️ HIGH WEATHER STRESS: Temperature and humidity "
                "are contributing strongly to City Stress.",
            )
        )
    elif weather_score >= 50:
        alerts.append(
            (
                "warning",
                "🌡️ MODERATE WEATHER STRESS: Weather conditions "
                "are contributing to City Stress.",
            )
        )

    if len(alerts) == 0:
        st.success(
            "🟢 City conditions are currently stable. "
            "No major alerts detected."
        )
    else:
        for alert_type, alert_message in alerts:
            if alert_type == "critical":
                st.error(alert_message)
            else:
                st.warning(alert_message)


    # ========================================================
    # NEXT 15-MINUTE PREDICTION
    # ========================================================

    prediction = None
    prediction_error = None

    if not using_historical_fallback:
        try:
            prediction = predict_next_stress(
                live_data=live_data
            )
        except Exception as exc:
            prediction_error = str(exc)

    # ========================================================
    # INTELLIGENT RECOMMENDATIONS
    # ========================================================

    st.subheader("🧠 Intelligent Recommendations")

    recommendations = generate_recommendations(
        city_stress_score=city_stress_score,
        pollution_score=pollution_score,
        congestion_ratio=congestion_ratio,
        weather_score=weather_score,
        predicted_stress=(
            prediction["predicted_stress"]
            if prediction is not None
            else None
        ),
    )

    for recommendation in recommendations:
        if recommendation.startswith("🔴"):
            st.error(recommendation)
        elif recommendation.startswith("🟠"):
            st.warning(recommendation)
        elif recommendation.startswith("🟢"):
            st.success(recommendation)
        else:
            st.info(recommendation)

    # ========================================================
    # PREDICTION DISPLAY
    # ========================================================

    st.subheader("🔮 Next 15-Minute Prediction")

    if prediction is not None:

        prediction_col1, prediction_col2, prediction_col3 = (
            st.columns(3)
        )

        with prediction_col1:
            st.metric(
                "Predicted City Stress",
                f"{prediction['predicted_stress']:.1f}",
            )

        with prediction_col2:

            predicted_level = prediction["stress_level"]

            if predicted_level == "Low":
                st.success(f"🟢 {predicted_level}")
            elif predicted_level == "Moderate":
                st.info(f"🟡 {predicted_level}")
            elif predicted_level == "High":
                st.warning(f"🟠 {predicted_level}")
            else:
                st.error(f"🔴 {predicted_level}")

        with prediction_col3:
            st.metric(
                "Model MAE",
                f"{prediction['mae']:.1f}",
            )

        st.caption(
            "Prediction horizon: 15 minutes | "
            f"Training samples: {prediction['training_samples']}"
        )

    elif prediction_error:
        st.warning(
            "⚠️ 15-minute prediction is temporarily unavailable."
        )
        st.caption(prediction_error)
    else:
        st.info(
            "ℹ️ Prediction is paused while using historical fallback data."
        )


    # ========================================================
    # ML FEATURE IMPORTANCE
    # ========================================================

    st.subheader("🧠 ML Feature Importance")

    try:
        feature_importance = get_feature_importance()

        if not feature_importance.empty:
            importance_chart = (
                feature_importance
                .set_index("Feature")["Importance"]
                .sort_values(ascending=True)
            )

            st.bar_chart(
                importance_chart,
                width="stretch",
            )

            st.caption(
                "Feature importance indicates how much each feature "
                "contributed to the Random Forest model's predictions. "
                "It does not establish causation."
            )
    except Exception as exc:
        st.warning(
            "Feature importance is temporarily unavailable."
        )
        st.caption(str(exc))


    # ========================================================
    # PREDICTION VS ACTUAL
    # ========================================================

    st.subheader("📊 Prediction vs Actual City Stress")

    try:
        comparison = get_prediction_vs_actual()

        if not comparison.empty:

            comparison_chart = comparison[
                [
                    "timestamp",
                    "actual_future_stress",
                    "predicted_future_stress",
                ]
            ].copy()

            comparison_chart["timestamp"] = pd.to_datetime(
                comparison_chart["timestamp"]
            )

            comparison_chart = comparison_chart.set_index(
                "timestamp"
            )

            comparison_chart = comparison_chart.rename(
                columns={
                    "actual_future_stress": "Actual Stress",
                    "predicted_future_stress": "Predicted Stress",
                }
            )

            st.line_chart(
                comparison_chart,
                width="stretch",
            )

            metric_col1, metric_col2, metric_col3 = st.columns(3)

            with metric_col1:
                st.metric(
                    "Comparison Samples",
                    len(comparison),
                )

            with metric_col2:
                st.metric(
                    "Prediction vs Actual MAE",
                    f"{comparison['absolute_error'].mean():.2f}",
                )

            with metric_col3:
                rmse = float(
                    np.sqrt(
                        np.mean(
                            comparison["prediction_error"] ** 2
                        )
                    )
                )

                st.metric(
                    "Prediction vs Actual RMSE",
                    f"{rmse:.2f}",
                )

            st.caption(
                "The chart compares the model's 15-minute predictions "
                "with observed City Stress values. With the current "
                "small historical sample, these comparison metrics "
                "should be treated as prototype diagnostics rather "
                "than production-level validation."
            )

    except Exception as exc:
        st.warning(
            "Prediction vs Actual analysis is temporarily unavailable."
        )
        st.caption(str(exc))


    # ========================================================
    # AIR QUALITY
    # ========================================================

    st.subheader("🌫️ Air Quality")

    air_col1, air_col2, air_col3 = st.columns(3)

    with air_col1:
        st.metric(
            "PM10",
            f"{pm10:.1f}",
        )

    with air_col2:
        st.metric(
            "US AQI",
            f"{us_aqi:.0f}",
        )

    with air_col3:
        st.metric(
            "Wind Speed",
            f"{wind_speed:.1f} km/h",
        )


    # ========================================================
    # TRAFFIC CONDITIONS
    # ========================================================

    st.subheader("🚗 Traffic Conditions")

    traffic_col1, traffic_col2, traffic_col3 = st.columns(3)

    with traffic_col1:
        st.metric(
            "Current Speed",
            f"{current_speed:.1f} km/h",
        )

    with traffic_col2:
        st.metric(
            "Free Flow Speed",
            f"{free_flow_speed:.1f} km/h",
        )

    with traffic_col3:
        st.metric(
            "Congestion",
            f"{congestion_ratio * 100:.1f}%",
        )


    # ========================================================
    # CITY STRESS COMPONENTS
    # ========================================================

    st.subheader("🏙️ City Stress Components")

    component_col1, component_col2, component_col3 = st.columns(3)

    with component_col1:
        st.metric(
            "Pollution Score",
            f"{pollution_score:.1f}",
        )

    with component_col2:
        st.metric(
            "Traffic Score",
            f"{traffic_score:.1f}",
        )

    with component_col3:
        st.metric(
            "Weather Stress",
            f"{weather_score:.1f}",
        )


    # ========================================================
    # CITY STRESS BREAKDOWN
    # ========================================================

    st.subheader("📊 City Stress Breakdown")

    breakdown = pd.Series(
        {
            "Pollution": pollution_score,
            "Traffic": traffic_score,
            "Weather": weather_score,
        }
    )

    st.bar_chart(
        breakdown,
        width="stretch",
    )


# ============================================================
# CITY INTELLIGENCE MAP
# ============================================================

st.subheader("🗺️ City Intelligence Map")

map_stress = (
    stress["city_stress_score"]
    if stress is not None
    else 0
)

map_level = (
    stress["stress_level"]
    if stress is not None
    else "Low"
)

city_map = folium.Map(
    location=[LATITUDE, LONGITUDE],
    zoom_start=12,
    tiles="OpenStreetMap",
)

folium.Circle(
    location=[LATITUDE, LONGITUDE],
    radius=5000,
    color="blue",
    fill=False,
    weight=2,
).add_to(city_map)

folium.Marker(
    [LATITUDE, LONGITUDE],
    tooltip="Coimbatore City Intelligence",
    popup=(
        f"City Stress: {map_stress:.1f}<br>"
        f"Stress Level: {map_level}"
    ),
    icon=folium.Icon(
        color=stress_color(map_level),
        icon="info-sign",
    ),
).add_to(city_map)

st_folium(
    city_map,
    width=1000,
    height=480,
)

st.caption(
    "📍 Map center represents the Coimbatore monitoring location. "
    "The marker color reflects the current City Stress level. "
    "The circle represents the 5 km monitoring zone."
)


# ============================================================
# HISTORICAL ANALYTICS
# ============================================================

if historical_data_available:

    # --------------------------------------------------------
    # HISTORICAL CITY STRESS TREND
    # --------------------------------------------------------

    st.subheader("📈 Historical City Stress Trend")

    stress_chart = make_chart_df(
        historical_data,
        ["city_stress_score"],
        {"city_stress_score": "City Stress"},
    )

    if not stress_chart.empty:
        stress_chart = stress_chart.reset_index()
        stress_chart = stress_chart.rename(columns={"index": "timestamp"})
        stress_chart = add_time_gap_breaks(stress_chart)

        fig = px.line(
            stress_chart,
            x="timestamp",
            y="City Stress",
            markers=True,
            title="Historical City Stress",
        )
        fig.update_yaxes(range=[0, 100])
        fig.update_xaxes(title="Time")
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")


    # --------------------------------------------------------
    # HISTORICAL PM2.5 TREND
    # --------------------------------------------------------

    st.subheader("🌫️ Historical PM2.5 Trend")

    pm_chart = make_chart_df(
        historical_data,
        ["pm2_5"],
        {"pm2_5": "PM2.5"},
    )

    if not pm_chart.empty:
        pm_chart = pm_chart.reset_index()
        pm_chart = pm_chart.rename(columns={"index": "timestamp"})
        pm_chart = add_time_gap_breaks(pm_chart)

        fig = px.line(
            pm_chart,
            x="timestamp",
            y="PM2.5",
            markers=True,
            title="Historical PM2.5",
        )
        fig.update_xaxes(title="Time")
        fig.update_yaxes(rangemode="tozero")
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")


    # --------------------------------------------------------
    # HISTORICAL TRAFFIC CONGESTION TREND
    # --------------------------------------------------------

    st.subheader("🚗 Historical Traffic Congestion Trend")

    congestion_chart = make_chart_df(
        historical_data,
        ["congestion_ratio"],
    )

    if not congestion_chart.empty:
        congestion_chart["congestion_ratio"] = (
            congestion_chart["congestion_ratio"] * 100
        )

        congestion_chart = congestion_chart.rename(
            columns={"congestion_ratio": "Congestion (%)"}
        )

        congestion_chart = congestion_chart.reset_index()
        congestion_chart = congestion_chart.rename(columns={"index": "timestamp"})
        congestion_chart = add_time_gap_breaks(congestion_chart)

        fig = px.line(
            congestion_chart,
            x="timestamp",
            y="Congestion (%)",
            markers=True,
            title="Historical Traffic Congestion",
        )
        fig.update_xaxes(title="Time")
        fig.update_yaxes(range=[0, 100])
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")


    # --------------------------------------------------------
    # HISTORICAL TEMPERATURE TREND
    # --------------------------------------------------------

    st.subheader("🌡️ Historical Temperature Trend")

    temperature_chart = make_chart_df(
        historical_data,
        ["temperature"],
        {"temperature": "Temperature (°C)"},
    )

    if not temperature_chart.empty:
        temperature_chart = temperature_chart.reset_index()
        temperature_chart = temperature_chart.rename(
            columns={"index": "timestamp"}
        )
        temperature_chart = add_time_gap_breaks(temperature_chart)

        fig = px.line(
            temperature_chart,
            x="timestamp",
            y="Temperature (°C)",
            markers=True,
            title="Historical Temperature",
        )
        fig.update_xaxes(title="Time")
        fig.update_yaxes(title="Temperature (°C)")
        fig.update_layout(height=430)
        st.plotly_chart(fig, width="stretch")


    # ========================================================
    # DATA QUALITY & MODEL HEALTH
    # ========================================================

    st.header("🛡️ Data Quality & Model Health")

    historical_count = len(historical_data)

    if "timestamp" in historical_data.columns:
        historical_timestamps = pd.to_datetime(
            historical_data["timestamp"], errors="coerce"
        ).dropna().sort_values()
    else:
        historical_timestamps = pd.Series(dtype="datetime64[ns]")

    if len(historical_timestamps) >= 2:
        observed_intervals = historical_timestamps.diff().dropna()
        exact_15_min_count = int(
            (observed_intervals == pd.Timedelta(minutes=15)).sum()
        )
        largest_gap = observed_intervals.max()
        largest_gap_minutes = largest_gap.total_seconds() / 60
    else:
        exact_15_min_count = 0
        largest_gap_minutes = 0

    # Use the prediction engine's actual training-pair definition when available.
    model_training_samples = None
    model_mae = None
    model_rmse = None

    if prediction is not None:
        model_training_samples = prediction.get("training_samples")
        model_mae = prediction.get("mae")
        model_rmse = prediction.get("rmse")

    # Completeness across the core historical variables.
    quality_columns = [
        column
        for column in [
            "timestamp",
            "temperature",
            "humidity",
            "wind_speed",
            "pm2_5",
            "pm10",
            "us_aqi",
            "current_speed",
            "free_flow_speed",
            "congestion_ratio",
            "city_stress_score",
        ]
        if column in historical_data.columns
    ]

    if quality_columns and historical_count > 0:
        total_cells = historical_count * len(quality_columns)
        missing_cells = int(historical_data[quality_columns].isna().sum().sum())
        completeness = max(
            0,
            min(100, (1 - missing_cells / total_cells) * 100),
        )
    else:
        completeness = 0

    health_col1, health_col2, health_col3, health_col4 = st.columns(4)

    with health_col1:
        st.metric("Historical Records", historical_count)

    with health_col2:
        st.metric("15-Min Training Pairs", model_training_samples or 0)

    with health_col3:
        st.metric("Data Completeness", f"{completeness:.1f}%")

    with health_col4:
        if model_mae is not None:
            st.metric("Model MAE", f"{model_mae:.2f}")
        else:
            st.metric("Model MAE", "N/A")

    if model_training_samples is not None and model_training_samples < 50:
        st.warning(
            f"⚠️ Small training sample: {model_training_samples} exact 15-minute "
            "pairs are currently available. Model results should be treated as "
            "prototype diagnostics until more historical data is collected."
        )
    elif model_training_samples is not None:
        st.success(
            f"🟢 Model training set currently contains "
            f"{model_training_samples} exact 15-minute pairs."
        )

    quality_col1, quality_col2 = st.columns(2)

    with quality_col1:
        st.markdown("**📡 Historical Data Status**")
        st.write(f"• Records available: **{historical_count}**")
        st.write(f"• Exact 15-minute intervals: **{exact_15_min_count}**")
        st.write(f"• Data completeness: **{completeness:.1f}%**")

        if largest_gap_minutes > 30:
            st.write(
                f"• Largest observed gap: **{largest_gap_minutes:.0f} minutes**"
            )
        else:
            st.write("• Largest observed gap: **≤ 30 minutes**")

        if not historical_timestamps.empty:
            latest_historical = historical_timestamps.max()
            st.write(
                f"• Latest historical record: "
                f"**{latest_historical.strftime('%d %b %Y, %I:%M %p')}**"
            )

    with quality_col2:
        st.markdown("**🤖 Model Health Status**")

        if model_training_samples is None:
            st.write("• Model status: **Unavailable**")
        else:
            st.write("• Model: **Random Forest Regressor**")
            st.write("• Prediction horizon: **15 minutes**")
            st.write(f"• Training pairs: **{model_training_samples}**")

            if model_mae is not None:
                st.write(f"• Test MAE: **{model_mae:.2f}**")
            if model_rmse is not None:
                st.write(f"• Test RMSE: **{model_rmse:.2f}**")

        st.caption(
            "Model metrics are prototype diagnostics because the current "
            "historical dataset is small."
        )

    st.divider()


    # ========================================================
    # ADVANCED CITY ANALYTICS
    # ========================================================

    st.subheader("📊 Advanced City Analytics")

    # --------------------------------------------------------
    # HISTORICAL SUMMARY
    # --------------------------------------------------------

    st.markdown("### 📋 Historical Summary")

    summary_columns = [
        "city_stress_score",
        "pm2_5",
        "temperature",
        "humidity",
        "congestion_ratio",
    ]

    available_summary_columns = [
        column
        for column in summary_columns
        if column in historical_data.columns
    ]

    summary_data = historical_data[
        available_summary_columns
    ].describe().T

    summary_data = summary_data.rename(
        columns={
            "count": "Count",
            "mean": "Mean",
            "std": "Std Dev",
            "min": "Minimum",
            "25%": "25th Percentile",
            "50%": "Median",
            "75%": "75th Percentile",
            "max": "Maximum",
        }
    )

    if "congestion_ratio" in summary_data.index:
        summary_data.loc[
            "congestion_ratio",
            [
                "Mean",
                "Minimum",
                "25th Percentile",
                "Median",
                "75th Percentile",
                "Maximum",
            ],
        ] *= 100

    st.dataframe(
        summary_data.round(2),
        width="stretch",
    )


    # --------------------------------------------------------
    # KEY ANALYTICS METRICS
    # --------------------------------------------------------

    analytics_col1, analytics_col2, analytics_col3, analytics_col4 = (
        st.columns(4)
    )

    with analytics_col1:
        st.metric(
            "Average City Stress",
            f"{historical_data['city_stress_score'].mean():.2f}",
        )

    with analytics_col2:
        st.metric(
            "Peak City Stress",
            f"{historical_data['city_stress_score'].max():.2f}",
        )

    with analytics_col3:
        st.metric(
            "Average PM2.5",
            f"{historical_data['pm2_5'].mean():.2f}",
        )

    with analytics_col4:
        st.metric(
            "Average Congestion",
            f"{historical_data['congestion_ratio'].mean() * 100:.2f}%",
        )


    # --------------------------------------------------------
    # STRESS DISTRIBUTION
    # --------------------------------------------------------

    st.markdown("### 📊 City Stress Distribution")

    stress_distribution = pd.cut(
        historical_data["city_stress_score"],
        bins=[-1, 25, 50, 75, 100],
        labels=[
            "Low",
            "Moderate",
            "High",
            "Critical",
        ],
    )

    stress_distribution_counts = (
        stress_distribution
        .value_counts()
        .reindex(
            [
                "Low",
                "Moderate",
                "High",
                "Critical",
            ],
            fill_value=0,
        )
    )

    st.bar_chart(
        stress_distribution_counts,
        width="stretch",
    )


    # --------------------------------------------------------
    # PEAK STRESS PERIODS
    # --------------------------------------------------------

    st.markdown("### 🔥 Peak City Stress Periods")

    peak_columns = [
        "timestamp",
        "city_stress_score",
        "pm2_5",
        "temperature",
        "humidity",
        "congestion_ratio",
    ]

    available_peak_columns = [
        column
        for column in peak_columns
        if column in historical_data.columns
    ]

    peak_stress = (
        historical_data[available_peak_columns]
        .sort_values(
            "city_stress_score",
            ascending=False,
        )
        .head(5)
        .copy()
    )

    if "congestion_ratio" in peak_stress.columns:
        peak_stress["congestion_ratio"] *= 100

    peak_stress = peak_stress.rename(
        columns={
            "timestamp": "Timestamp",
            "city_stress_score": "City Stress",
            "pm2_5": "PM2.5",
            "temperature": "Temperature",
            "humidity": "Humidity",
            "congestion_ratio": "Congestion (%)",
        }
    )

    st.dataframe(
        peak_stress.round(2),
        width="stretch",
    )


    # --------------------------------------------------------
    # CORRELATION ANALYSIS
    # --------------------------------------------------------

    st.markdown("### 🔗 City Stress Correlation Analysis")

    correlation_columns = [
        "city_stress_score",
        "pm2_5",
        "pm10",
        "temperature",
        "humidity",
        "wind_speed",
        "us_aqi",
        "congestion_ratio",
    ]

    available_correlation_columns = [
        column
        for column in correlation_columns
        if column in historical_data.columns
    ]

    correlation_matrix = (
        historical_data[
            available_correlation_columns
        ].corr()
    )

    if "city_stress_score" in correlation_matrix.columns:

        stress_correlations = (
            correlation_matrix[
                "city_stress_score"
            ]
            .drop(
                "city_stress_score"
            )
            .sort_values(
                ascending=False
            )
            .to_frame(
                name="Correlation with City Stress"
            )
        )

        st.dataframe(
            stress_correlations.round(3),
            width="stretch",
        )

        st.caption(
            "Correlation measures statistical association in the "
            "historical sample; it does not establish causation."
        )


    # --------------------------------------------------------
    # PROFESSIONAL SCATTER ANALYTICS
    # --------------------------------------------------------

    st.markdown("### 🌫️ Pollution vs City Stress")

    pollution_analysis = historical_data[
        [
            "timestamp",
            "pm2_5",
            "city_stress_score",
        ]
    ].dropna().copy()

    pollution_analysis = pollution_analysis.rename(
        columns={
            "pm2_5": "PM2.5",
            "city_stress_score": "City Stress",
        }
    )

    if not pollution_analysis.empty:

        fig_pollution = px.scatter(
            pollution_analysis,
            x="PM2.5",
            y="City Stress",
            hover_data=["timestamp"],
            labels={
                "PM2.5": "PM2.5 Concentration",
                "City Stress": "City Stress Score",
            },
            title="Relationship Between PM2.5 and City Stress",
            trendline="ols",
        )

        fig_pollution.update_traces(
            marker=dict(size=10),
        )

        fig_pollution.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=70, b=20),
            xaxis=dict(
                title="PM2.5",
                rangemode="tozero",
            ),
            yaxis=dict(
                title="City Stress",
                range=[0, 100],
            ),
        )

        st.plotly_chart(
            fig_pollution,
            width="stretch",
        )


    # --------------------------------------------------------
    # TRAFFIC VS CITY STRESS
    # --------------------------------------------------------

    st.markdown("### 🚗 Traffic vs City Stress")

    traffic_analysis = historical_data[
        [
            "timestamp",
            "congestion_ratio",
            "city_stress_score",
        ]
    ].dropna().copy()

    traffic_analysis["Congestion (%)"] = (
        traffic_analysis["congestion_ratio"] * 100
    )

    traffic_analysis = traffic_analysis.rename(
        columns={
            "city_stress_score": "City Stress",
        }
    )

    if not traffic_analysis.empty:

        fig_traffic = px.scatter(
            traffic_analysis,
            x="Congestion (%)",
            y="City Stress",
            hover_data=["timestamp"],
            labels={
                "Congestion (%)": "Traffic Congestion (%)",
                "City Stress": "City Stress Score",
            },
            title="Relationship Between Traffic Congestion and City Stress",
            trendline="ols",
        )

        fig_traffic.update_traces(
            marker=dict(size=10),
        )

        fig_traffic.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=70, b=20),
            xaxis=dict(
                title="Traffic Congestion (%)",
                range=[0, 100],
            ),
            yaxis=dict(
                title="City Stress",
                range=[0, 100],
            ),
        )

        st.plotly_chart(
            fig_traffic,
            width="stretch",
        )


    # --------------------------------------------------------
    # TEMPERATURE VS CITY STRESS
    # --------------------------------------------------------

    st.markdown("### 🌡️ Temperature vs City Stress")

    weather_analysis = historical_data[
        [
            "timestamp",
            "temperature",
            "city_stress_score",
        ]
    ].dropna().copy()

    weather_analysis = weather_analysis.rename(
        columns={
            "temperature": "Temperature",
            "city_stress_score": "City Stress",
        }
    )

    if not weather_analysis.empty:

        fig_temperature = px.scatter(
            weather_analysis,
            x="Temperature",
            y="City Stress",
            hover_data=["timestamp"],
            labels={
                "Temperature": "Temperature (°C)",
                "City Stress": "City Stress Score",
            },
            title="Relationship Between Temperature and City Stress",
            trendline="ols",
        )

        fig_temperature.update_traces(
            marker=dict(size=10),
        )

        temperature_min = max(
            0,
            float(weather_analysis["Temperature"].min()) - 1,
        )

        temperature_max = (
            float(weather_analysis["Temperature"].max()) + 1
        )

        fig_temperature.update_layout(
            height=500,
            margin=dict(l=20, r=20, t=70, b=20),
            xaxis=dict(
                title="Temperature (°C)",
                range=[
                    temperature_min,
                    temperature_max,
                ],
            ),
            yaxis=dict(
                title="City Stress",
                range=[0, 100],
            ),
        )

        st.plotly_chart(
            fig_temperature,
            width="stretch",
        )


    # --------------------------------------------------------
    # ANALYTICS INTERPRETATION
    # --------------------------------------------------------

    st.markdown("### 💡 Analytics Interpretation")

    average_stress = historical_data[
        "city_stress_score"
    ].mean()

    peak_stress = historical_data[
        "city_stress_score"
    ].max()

    high_or_critical_count = int(
        (
            historical_data["city_stress_score"] >= 50
        ).sum()
    )

    total_rows = len(historical_data)

    if high_or_critical_count > 0:
        stress_observation = (
            f"{high_or_critical_count} of {total_rows} historical "
            "records reached High or Critical stress."
        )
    else:
        stress_observation = (
            "No historical record reached High or Critical stress."
        )

    st.info(
        f"📌 Average historical City Stress is "
        f"{average_stress:.2f}, with a peak of {peak_stress:.2f}. "
        f"{stress_observation}"
    )


else:

    st.warning(
        "⚠️ Historical dashboard data is not available. "
        "Historical charts and advanced analytics are unavailable."
    )


# ============================================================
# LIVE DATA INFORMATION
# ============================================================

st.subheader("ℹ️ Live Data Information")

if live_data is not None:

    raw_timestamp = live_data.get("timestamp")

    try:
        display_timestamp = pd.to_datetime(
            raw_timestamp
        ).strftime(
            "%d %b %Y, %I:%M %p"
        )
    except Exception:
        display_timestamp = str(raw_timestamp)

    if using_historical_fallback:
        st.write(
            f"**Latest available timestamp:** "
            f"{display_timestamp}"
        )
        st.caption(
            "Currently displaying historical fallback data because "
            "the live API request was temporarily unavailable."
        )
    else:
        st.write(
            f"**Latest API timestamp:** "
            f"{display_timestamp}"
        )

    st.write(
        "**Data sources:** Weather API, Air Quality API, "
        "TomTom Traffic API"
    )


# ============================================================
# METHODOLOGY
# ============================================================

with st.expander("🔍 View Live Calculation Details"):

    st.markdown(
        """
### City Stress Formula

**City Stress Index**

- Pollution = 40%
- Traffic = 35%
- Weather = 25%

**Weather Stress**

- Heat = 70%
- Humidity = 30%

**Stress Levels**

- 0–24.99 → Low
- 25–49.99 → Moderate
- 50–74.99 → High
- 75–100 → Critical

### Important Prototype Notes

The City Stress weights and min-max normalization ranges are
project-defined prototype methodology, not scientifically validated
thresholds.

The current ML model is trained on a small historical sample and
should be treated as a proof-of-concept predictive system rather than
a production forecasting model.

Correlation and Random Forest feature importance describe statistical
relationships/model behavior; they do not prove causation.
"""
    )


# ============================================================
# FOOTER
# ============================================================

st.markdown("---")

st.caption(
    "Coimbatore City Intelligence — "
    "Real-time Data Analysis & Predictive Analytics Project"
)
