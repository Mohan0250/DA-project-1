"""
Intelligent Recommendation Engine
Coimbatore City Intelligence System

Generates actionable recommendations from:
- City Stress
- Pollution
- Traffic
- Weather
- ML prediction
"""


def generate_recommendations(
    city_stress_score,
    pollution_score,
    congestion_ratio,
    weather_score,
    predicted_stress=None
):
    recommendations = []

    # ========================================================
    # OVERALL CITY STRESS
    # ========================================================

    if city_stress_score >= 75:
        recommendations.append(
            "🔴 CRITICAL PRIORITY | Overall City Stress is critical. "
            "Recommended Action: Initiate immediate city-level monitoring "
            "and coordinated intervention."
        )

    elif city_stress_score >= 50:
        recommendations.append(
            "🟠 HIGH PRIORITY | Overall City Stress is high. "
            "Recommended Action: Increase city-wide environmental and "
            "traffic monitoring."
        )

    elif city_stress_score >= 25:
        recommendations.append(
            "🟡 MODERATE PRIORITY | Overall City Stress is moderate. "
            "Recommended Action: Continue monitoring environmental and "
            "traffic conditions."
        )

    else:
        recommendations.append(
            "🟢 STABLE | Overall City Stress is low. "
            "Recommended Action: Continue routine city monitoring."
        )


    # ========================================================
    # POLLUTION
    # ========================================================

    if pollution_score >= 75:
        recommendations.append(
            "🔴 POLLUTION ALERT | Pollution is strongly contributing "
            "to City Stress. "
            "Recommended Action: Increase air-quality monitoring and "
            "consider pollution advisories."
        )

    elif pollution_score >= 50:
        recommendations.append(
            "🟠 POLLUTION ATTENTION | Air pollution is elevated. "
            "Recommended Action: Increase air-quality monitoring and "
            "track PM2.5 trends."
        )

    elif pollution_score >= 25:
        recommendations.append(
            "🟡 POLLUTION WATCH | Pollution is contributing moderately "
            "to City Stress. "
            "Recommended Action: Continue monitoring PM2.5 conditions."
        )


    # ========================================================
    # TRAFFIC
    # ========================================================

    if congestion_ratio >= 0.75:
        recommendations.append(
            "🔴 TRAFFIC ALERT | Severe congestion detected. "
            "Recommended Action: Prioritize traffic-management attention "
            "and monitor major corridors."
        )

    elif congestion_ratio >= 0.50:
        recommendations.append(
            "🟠 TRAFFIC ATTENTION | Traffic congestion is elevated. "
            "Recommended Action: Monitor congested corridors and traffic "
            "flow conditions."
        )

    elif congestion_ratio >= 0.25:
        recommendations.append(
            "🟡 TRAFFIC WATCH | Moderate congestion detected. "
            "Recommended Action: Continue monitoring traffic flow."
        )


    # ========================================================
    # WEATHER
    # ========================================================

    if weather_score >= 75:
        recommendations.append(
            "🔴 WEATHER ALERT | Weather conditions are strongly "
            "contributing to City Stress. "
            "Recommended Action: Consider heat/weather-risk advisories "
            "and increased monitoring."
        )

    elif weather_score >= 50:
        recommendations.append(
            "🟠 WEATHER ATTENTION | Weather conditions are contributing "
            "significantly to City Stress. "
            "Recommended Action: Continue monitoring temperature and "
            "humidity conditions."
        )

    elif weather_score >= 25:
        recommendations.append(
            "🟡 WEATHER WATCH | Weather conditions are contributing "
            "moderately to City Stress. "
            "Recommended Action: Continue routine weather monitoring."
        )


    # ========================================================
    # ML EARLY WARNING
    # ========================================================

    if predicted_stress is not None:

        stress_difference = (
            predicted_stress - city_stress_score
        )

        if predicted_stress >= 75:

            recommendations.append(
                "🔴 ML EARLY WARNING | Predicted City Stress is critical "
                "for the next 15 minutes. "
                "Recommended Action: Prepare for immediate monitoring "
                "and intervention."
            )

        elif predicted_stress >= 50:

            recommendations.append(
                "🟠 ML EARLY WARNING | Predicted City Stress is high "
                "for the next 15 minutes. "
                "Recommended Action: Increase monitoring of pollution, "
                "traffic, and weather conditions."
            )

        elif stress_difference >= 10:

            recommendations.append(
                "🟡 ML RISING TREND | City Stress is predicted to rise "
                "significantly within the next 15 minutes. "
                "Recommended Action: Monitor conditions closely for "
                "an emerging stress event."
            )

        elif stress_difference >= 5:

            recommendations.append(
                "🔵 ML WATCH | City Stress is predicted to increase "
                "within the next 15 minutes. "
                "Recommended Action: Continue close monitoring."
            )

        elif stress_difference <= -10:

            recommendations.append(
                "🟢 ML IMPROVEMENT | City Stress is predicted to decrease "
                "significantly within the next 15 minutes. "
                "Recommended Action: Continue monitoring as conditions "
                "appear to be improving."
            )


    return recommendations