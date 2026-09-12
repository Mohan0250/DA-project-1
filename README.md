# Coimbatore City Intelligence & Predictive Analytics

A Data Science and Machine Learning project for monitoring and predicting urban stress using real-time weather, air quality, and traffic data.

## Project Overview

The Coimbatore City Intelligence System combines live environmental and traffic data with historical analytics and machine learning to calculate a City Stress Score and predict short-term changes in urban conditions.

## Key Features

- Real-time weather monitoring
- Real-time air-quality monitoring
- Real-time traffic monitoring
- City Stress Index
- Intelligent city alerts
- Action-based recommendations
- Random Forest prediction model
- 15-minute City Stress prediction
- Historical trend analysis
- Correlation analysis
- Pollution, traffic and temperature relationship analysis
- City Intelligence Map
- Data Quality and Model Health monitoring

## City Stress Methodology

City Stress is calculated using three major components:

- Pollution: 40%
- Traffic: 35%
- Weather: 25%

Weather Stress is calculated using:

- Heat: 70%
- Humidity: 30%

### Stress Levels

| Score | Level |
|---|---|
| 0-24.99 | Low |
| 25-49.99 | Moderate |
| 50-74.99 | High |
| 75-100 | Critical |

## Machine Learning

The predictive component uses a Random Forest Regressor to estimate City Stress approximately 15 minutes into the future.

### Model Features

- Temperature
- Humidity
- Wind Speed
- PM2.5
- PM10
- US AQI
- Current Speed
- Congestion Ratio

The current model is a proof-of-concept because the historical dataset is small.

## Project Structure

`	ext
DA project-1/
|
+-- dashboard/
|   +-- city_intelligence_dashboard.py
|   +-- live_data_pipeline.py
|   +-- predictive_engine.py
|   +-- recommendation_engine.py
|   +-- update_dashboard_data.py
|   +-- update_dashboard.bat
|
+-- data/
+-- collectors/
+-- notebooks/
+-- config/
|
+-- requirements.txt
+-- README.md
+-- .gitignore
