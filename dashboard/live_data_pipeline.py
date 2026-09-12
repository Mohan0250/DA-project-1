import requests


# ============================================================
# CITY LOCATION
# ============================================================

LATITUDE = 11.0168
LONGITUDE = 76.9558


# ============================================================
# API URLs
# ============================================================

WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

TRAFFIC_URL = (
    "https://api.tomtom.com/traffic/services/"
    "4/flowSegmentData/absolute/10/json"
)


# ============================================================
# NORMALIZATION SETTINGS
# ============================================================

PM25_MIN = 5
PM25_MAX = 25

TEMP_MIN = 20
TEMP_MAX = 40

HUMIDITY_MIN = 20
HUMIDITY_MAX = 90


# ============================================================
# MAIN LIVE DATA FUNCTION
# ============================================================

def get_live_city_data():

    # --------------------------------------------------------
    # 1. WEATHER API
    # --------------------------------------------------------

    weather_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": [
            "temperature_2m",
            "relative_humidity_2m",
            "precipitation",
            "wind_speed_10m"
        ],
        "timezone": "Asia/Kolkata"
    }

    weather_response = requests.get(
        WEATHER_URL,
        params=weather_params,
        timeout=30
    )

    weather_response.raise_for_status()

    weather_data = weather_response.json()

    weather = weather_data["current"]


    # --------------------------------------------------------
    # 2. AIR QUALITY API
    # --------------------------------------------------------

    air_quality_params = {
        "latitude": LATITUDE,
        "longitude": LONGITUDE,
        "current": [
            "pm2_5",
            "pm10",
            "carbon_monoxide",
            "nitrogen_dioxide",
            "sulphur_dioxide",
            "ozone",
            "us_aqi"
        ],
        "timezone": "Asia/Kolkata"
    }

    air_response = requests.get(
        AIR_QUALITY_URL,
        params=air_quality_params,
        timeout=30
    )

    air_response.raise_for_status()

    air_data = air_response.json()

    air = air_data["current"]


    # --------------------------------------------------------
    # 3. TOMTOM TRAFFIC API
    # --------------------------------------------------------

    with open(
        r"C:\Users\mohan\tomtom_key.txt",
        "r"
    ) as file:

        tomtom_key = file.read().strip()


    traffic_params = {
        "key": tomtom_key,
        "point": f"{LATITUDE},{LONGITUDE}",
        "unit": "KMPH"
    }

    traffic_response = requests.get(
        TRAFFIC_URL,
        params=traffic_params,
        timeout=30
    )

    traffic_response.raise_for_status()

    traffic_data = traffic_response.json()

    traffic = traffic_data["flowSegmentData"]


    # --------------------------------------------------------
    # 4. CONGESTION SCORE
    # --------------------------------------------------------

    if traffic["freeFlowSpeed"] == 0:

        congestion_ratio = 0

    else:

        congestion_ratio = (
            1
            - (
                traffic["currentSpeed"]
                / traffic["freeFlowSpeed"]
            )
        )

    congestion_ratio = max(
        0,
        min(1, congestion_ratio)
    )


    # --------------------------------------------------------
    # 5. POLLUTION SCORE
    # --------------------------------------------------------

    pm25 = air["pm2_5"]

    pollution_score = (
        (pm25 - PM25_MIN)
        / (PM25_MAX - PM25_MIN)
    )

    pollution_score = max(
        0,
        min(1, pollution_score)
    )


    # --------------------------------------------------------
    # 6. WEATHER STRESS
    # --------------------------------------------------------

    temperature = weather["temperature_2m"]

    humidity = weather["relative_humidity_2m"]


    # Heat score

    heat_score = (
        (temperature - TEMP_MIN)
        / (TEMP_MAX - TEMP_MIN)
    )

    heat_score = max(
        0,
        min(1, heat_score)
    )


    # Humidity score

    humidity_score = (
        (humidity - HUMIDITY_MIN)
        / (HUMIDITY_MAX - HUMIDITY_MIN)
    )

    humidity_score = max(
        0,
        min(1, humidity_score)
    )


    # Weather stress

    weather_stress_score = (
        0.7 * heat_score
        + 0.3 * humidity_score
    )


    # --------------------------------------------------------
    # 7. CITY STRESS INDEX
    # --------------------------------------------------------

    city_stress_index = (
        0.40 * pollution_score
        + 0.35 * congestion_ratio
        + 0.25 * weather_stress_score
    )


    city_stress_score = (
        city_stress_index * 100
    )


    # --------------------------------------------------------
    # 8. STRESS LEVEL
    # --------------------------------------------------------

    if city_stress_score <= 25:

        stress_level = "Low"

    elif city_stress_score <= 50:

        stress_level = "Moderate"

    elif city_stress_score <= 75:

        stress_level = "High"

    else:

        stress_level = "Critical"


    # --------------------------------------------------------
    # 9. RETURN LIVE DATA
    # --------------------------------------------------------

    return {

        "timestamp": weather["time"],

        "temperature": temperature,

        "humidity": humidity,

        "precipitation": weather["precipitation"],

        "wind_speed": weather["wind_speed_10m"],

        "pm2_5": air["pm2_5"],

        "pm10": air["pm10"],

        "carbon_monoxide": air["carbon_monoxide"],

        "nitrogen_dioxide": air["nitrogen_dioxide"],

        "sulphur_dioxide": air["sulphur_dioxide"],

        "ozone": air["ozone"],

        "us_aqi": air["us_aqi"],

        "current_speed": traffic["currentSpeed"],

        "free_flow_speed": traffic["freeFlowSpeed"],

        "current_travel_time": traffic["currentTravelTime"],

        "free_flow_travel_time": traffic["freeFlowTravelTime"],

        "confidence": traffic["confidence"],

        "road_closure": traffic["roadClosure"],

        "congestion_ratio": congestion_ratio,

        "pollution_score": pollution_score,

        "heat_score": heat_score,

        "humidity_score": humidity_score,

        "weather_stress_score": weather_stress_score,

        "city_stress_index": city_stress_index,

        "city_stress_score": city_stress_score,

        "stress_level": stress_level
    }


# ============================================================
# DIRECT TEST
# ============================================================

if __name__ == "__main__":

    live = get_live_city_data()

    print()
    print("==========================================")
    print("LIVE CITY INTELLIGENCE")
    print("==========================================")

    print(
        "Timestamp:",
        live["timestamp"]
    )

    print(
        "Temperature:",
        live["temperature"],
        "°C"
    )

    print(
        "Humidity:",
        live["humidity"],
        "%"
    )

    print(
        "PM2.5:",
        live["pm2_5"]
    )

    print(
        "US AQI:",
        live["us_aqi"]
    )

    print(
        "Traffic Speed:",
        live["current_speed"],
        "km/h"
    )

    print(
        "Congestion:",
        round(
            live["congestion_ratio"] * 100,
            1
        ),
        "%"
    )

    print(
        "City Stress Score:",
        round(
            live["city_stress_score"],
            2
        )
    )

    print(
        "Stress Level:",
        live["stress_level"]
    )

    print("==========================================")