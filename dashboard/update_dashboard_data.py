import pandas as pd


# ============================================================
# 1. LOAD RAW DATA
# ============================================================

weather = pd.read_csv(
    r"C:\Users\mohan\weather_data.csv"
)

air = pd.read_csv(
    r"C:\Users\mohan\air_quality_data.csv"
)

traffic = pd.read_csv(
    r"C:\Users\mohan\traffic_data.csv"
)


# ============================================================
# 2. CONVERT TIMESTAMPS
# ============================================================

weather["timestamp"] = pd.to_datetime(
    weather["timestamp"]
)

air["timestamp"] = pd.to_datetime(
    air["timestamp"]
)

traffic["timestamp"] = pd.to_datetime(
    traffic["timestamp"]
)


# ============================================================
# 3. SORT DATA
# ============================================================

weather = weather.sort_values("timestamp")
air = air.sort_values("timestamp")
traffic = traffic.sort_values("timestamp")


# ============================================================
# 4. MERGE WEATHER + AIR QUALITY
# ============================================================

weather_air = pd.merge_asof(
    weather,
    air,
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("30min")
)


# ============================================================
# 5. MERGE WEATHER + AIR + TRAFFIC
# ============================================================

combined = pd.merge_asof(
    weather_air,
    traffic,
    on="timestamp",
    direction="nearest",
    tolerance=pd.Timedelta("30min")
)


# ============================================================
# 6. REMOVE INCOMPLETE RECORDS
# ============================================================

combined_clean = combined.dropna(
    subset=[
        "temperature",
        "humidity",
        "pm2_5",
        "pm10",
        "us_aqi",
        "current_speed",
        "free_flow_speed"
    ]
).copy()


# ============================================================
# 7. CALCULATE CONGESTION
# ============================================================

combined_clean["congestion_ratio"] = (
    1 - (
        combined_clean["current_speed"]
        / combined_clean["free_flow_speed"]
    )
)


# ============================================================
# 8. CALCULATE POLLUTION SCORE
# ============================================================

pm25_min = combined_clean["pm2_5"].min()
pm25_max = combined_clean["pm2_5"].max()

if pm25_max == pm25_min:

    combined_clean["pollution_score"] = 0

else:

    combined_clean["pollution_score"] = (
        (combined_clean["pm2_5"] - pm25_min)
        / (pm25_max - pm25_min)
    )


# ============================================================
# 9. CALCULATE WEATHER STRESS
# ============================================================

temp_min = combined_clean["temperature"].min()
temp_max = combined_clean["temperature"].max()

humidity_min = combined_clean["humidity"].min()
humidity_max = combined_clean["humidity"].max()


# Heat score

if temp_max == temp_min:

    combined_clean["heat_score"] = 0

else:

    combined_clean["heat_score"] = (
        (combined_clean["temperature"] - temp_min)
        / (temp_max - temp_min)
    )


# Humidity score

if humidity_max == humidity_min:

    combined_clean["humidity_score"] = 0

else:

    combined_clean["humidity_score"] = (
        (combined_clean["humidity"] - humidity_min)
        / (humidity_max - humidity_min)
    )


# Combined weather stress

combined_clean["weather_stress_score"] = (
    0.7 * combined_clean["heat_score"]
    + 0.3 * combined_clean["humidity_score"]
)


# ============================================================
# 10. CALCULATE CITY STRESS INDEX
# ============================================================

combined_clean["city_stress_index"] = (
    0.40 * combined_clean["pollution_score"]
    + 0.35 * combined_clean["congestion_ratio"]
    + 0.25 * combined_clean["weather_stress_score"]
)


# ============================================================
# 11. CONVERT INDEX TO 0-100 SCORE
# ============================================================

combined_clean["city_stress_score"] = (
    combined_clean["city_stress_index"] * 100
)


# ============================================================
# 12. ASSIGN STRESS LEVEL
# ============================================================

combined_clean["stress_level"] = pd.cut(
    combined_clean["city_stress_score"],
    bins=[-1, 25, 50, 75, 100],
    labels=[
        "Low",
        "Moderate",
        "High",
        "Critical"
    ]
)


# ============================================================
# 13. PREPARE DASHBOARD DATA
# ============================================================

dashboard_data = combined_clean[
    [
        "timestamp",
        "temperature",
        "humidity",
        "precipitation",
        "wind_speed",
        "pm2_5",
        "pm10",
        "us_aqi",
        "current_speed",
        "free_flow_speed",
        "congestion_ratio",
        "city_stress_score",
        "stress_level"
    ]
].copy()


# ============================================================
# 14. SAVE DASHBOARD DATA
# ============================================================

dashboard_data.to_csv(
    r"C:\Users\mohan\dashboard_data.csv",
    index=False
)


# ============================================================
# 15. DISPLAY UPDATE INFORMATION
# ============================================================

print()
print("==========================================")
print("Dashboard data updated successfully.")
print("==========================================")
print("Weather records:", len(weather))
print("Air quality records:", len(air))
print("Traffic records:", len(traffic))
print("Combined records:", len(combined_clean))
print(
    "Latest timestamp:",
    dashboard_data["timestamp"].max()
)
print(
    "Latest city stress:",
    round(
        dashboard_data["city_stress_score"].iloc[-1],
        2
    )
)
print(
    "Latest stress level:",
    dashboard_data["stress_level"].iloc[-1]
)
print("==========================================")