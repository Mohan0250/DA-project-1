# Coimbatore City Intelligence & Predictive Analytics

A Data Science and Machine Learning project for monitoring, analyzing, and predicting urban stress using real-time weather, air quality, and traffic data.

---

## Project Overview

The **Coimbatore City Intelligence System** combines live environmental and traffic data with historical analytics and machine learning to calculate a **City Stress Score** and predict short-term changes in urban conditions.

The system transforms real-time city data into meaningful insights through an interactive Streamlit dashboard.

---

## Key Features

### Real-Time Monitoring

- Real-time weather monitoring
- Real-time air-quality monitoring
- Real-time traffic monitoring
- Live API status monitoring

### City Intelligence

- City Stress Index
- Intelligent city alerts
- Action-based recommendations
- City Stress component breakdown
- Interactive City Intelligence Map

### Data Science & Analytics

- Historical trend analysis
- Correlation analysis
- Pollution relationship analysis
- Traffic relationship analysis
- Temperature relationship analysis
- Data quality monitoring

### Machine Learning

- Random Forest prediction model
- 15-minute City Stress prediction
- Prediction vs Actual analysis
- ML feature importance analysis
- Model performance metrics
- Model health monitoring

---

## System Architecture

```text
                    REAL-TIME DATA SOURCES
                             |
              +--------------+--------------+
              |              |              |
           Weather       Air Quality      Traffic
              |              |              |
              +--------------+--------------+
                             |
                             v
                   Live Data Pipeline
                             |
                             v
                    City Stress Engine
                             |
                +------------+------------+
                |                         |
                v                         v
        Streamlit Dashboard       Machine Learning
                |                         |
                |                         v
                |                 15-Minute Prediction
                |                         |
                +------------+------------+
                             |
                             v
                 City Intelligence Output
```

---

## City Stress Methodology

The **City Stress Score** is calculated using three major components:

| Component | Weight |
|---|---:|
| Pollution | 40% |
| Traffic | 35% |
| Weather | 25% |

### Weather Stress

Weather Stress is calculated using:

| Weather Component | Weight |
|---|---:|
| Heat | 70% |
| Humidity | 30% |

### Stress Levels

| Score | Level |
|---:|---|
| 0 - 24.99 | Low |
| 25 - 49.99 | Moderate |
| 50 - 74.99 | High |
| 75 - 100 | Critical |

> **Note:** The weights and normalization ranges used by this project are project-defined prototype methodology and are not intended to represent an officially validated urban stress standard.

---

## Intelligent City Alerts

The system analyzes current urban conditions and generates alerts based on factors such as:

- Overall city stress
- Air pollution
- Traffic congestion
- Weather conditions
- Predicted changes in City Stress

This allows users to quickly identify conditions that may require closer monitoring.

---

## Intelligent Recommendations

The recommendation engine converts detected conditions into action-oriented messages.

Examples include:

- Pollution monitoring recommendations
- Traffic monitoring recommendations
- Weather condition alerts
- Overall city stress monitoring
- Machine-learning-based prediction alerts

This provides a simple **decision-support layer** on top of the analytical results.

---

## Machine Learning

The predictive component uses a **Random Forest Regressor** to estimate City Stress approximately **15 minutes into the future**.

### Model Features

The model uses the following features:

- Temperature
- Humidity
- Wind Speed
- PM2.5
- PM10
- US AQI
- Current Speed
- Congestion Ratio

### Machine Learning Workflow

```text
Historical Data
      |
      v
Data Preparation
      |
      v
Feature Selection
      |
      v
Time-Ordered Training Data
      |
      v
Random Forest Regressor
      |
      v
15-Minute City Stress Prediction
      |
      v
Prediction + Model Evaluation
```

### Model Evaluation

The predictive system evaluates model performance using:

- Mean Absolute Error (MAE)
- Root Mean Squared Error (RMSE)
- Feature Importance
- Prediction vs Actual analysis

The dashboard also displays information about:

- Training samples
- Data completeness
- Historical time gaps
- Model health

> **Note:** The current model is a proof-of-concept because the available historical dataset is relatively small. Predictions and metrics should therefore be treated as prototype results rather than production-level forecasts.

---

## Data Analysis

The dashboard provides historical analysis of:

- City Stress
- PM2.5
- Traffic Congestion
- Temperature

It also provides analytical views for examining relationships between:

- Pollution and City Stress
- Traffic and City Stress
- Temperature and City Stress
- Individual City Stress components

Correlation analysis is used to identify statistical relationships within the available historical data.

> **Important:** Correlation and feature importance indicate statistical relationships or model behavior. They do not establish causation.

---

## City Intelligence Map

The system includes an interactive map centered on **Coimbatore**.

The map provides:

- Geographic visualization
- Current City Stress indication
- Stress-level information
- Interactive map navigation

---

## Dashboard

The Streamlit dashboard integrates the complete system into a single interface.

### Dashboard Sections

- Live API Status
- Latest City Conditions
- Current Stress Level
- Intelligent City Alerts
- Intelligent Recommendations
- Next 15-Minute Prediction
- ML Feature Importance
- Prediction vs Actual
- Air Quality
- Traffic Conditions
- City Stress Components
- City Stress Breakdown
- City Intelligence Map
- Historical City Stress Trend
- Historical PM2.5 Trend
- Historical Traffic Congestion Trend
- Historical Temperature Trend
- Advanced City Analytics
- Data Quality & Model Health
- Methodology

---

## Technology Stack

### Programming

- Python

### Data Science & Machine Learning

- Pandas
- NumPy
- Scikit-learn

### Visualization

- Plotly
- Folium

### Dashboard

- Streamlit
- Streamlit-Folium

### APIs

- Open-Meteo Weather API
- Open-Meteo Air Quality API
- TomTom Traffic API

---

## Project Structure

```text
DA-project-1/
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
```

---

## Installation

### 1. Clone the Repository

```bash
git clone https://github.com/Mohan0250/DA-project-1.git
cd DA-project-1
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Run the Dashboard

```bash
streamlit run dashboard/city_intelligence_dashboard.py
```

The Streamlit dashboard will start locally in your browser.

---

## API Configuration

The project uses external APIs to collect real-time city data.

API keys and credentials are intentionally **not included in this repository**.

Sensitive files are excluded using `.gitignore`.

### Security

Never commit the following to GitHub:

- API keys
- Passwords
- Authentication tokens
- `.env` files
- Private datasets
- Personal credentials

---

## Data Quality & Model Health

The dashboard includes a dedicated **Data Quality & Model Health** section.

It monitors:

- Historical record count
- Number of training pairs
- Data completeness
- Largest historical time gap
- Latest historical timestamp
- Model MAE
- Model RMSE
- Training sample warnings
- Random Forest model status

This helps communicate the reliability and limitations of the current analytical pipeline.

---

## Project Limitations

The current implementation has several limitations:

1. The historical dataset is relatively small.
2. The machine-learning model is a proof-of-concept.
3. City Stress weights are project-defined.
4. Normalization ranges are prototype assumptions.
5. Real-time API availability can affect dashboard updates.
6. Predictions should not be interpreted as production-level forecasts.
7. Correlation does not imply causation.

---

## Future Improvements

Possible future improvements include:

- Larger historical datasets
- Long-term data collection
- Advanced time-series forecasting
- LSTM-based forecasting
- Transformer-based forecasting
- Automated anomaly detection
- Support for multiple cities
- Real-time traffic heatmaps
- More detailed weather forecasting
- Population and mobility data integration
- Event-based city stress analysis
- Cloud deployment
- Automated model retraining
- Production-grade API management
- Continuous model performance monitoring

---

## Project Goal

The goal of this project is to demonstrate how **Data Science, Machine Learning, real-time APIs, and interactive visualization** can be combined to create an intelligent urban monitoring and short-term predictive analytics system.

The overall workflow is:

```text
Raw Data
   |
   v
Data Collection
   |
   v
Data Processing
   |
   v
City Stress Analysis
   |
   v
Machine Learning
   |
   v
Prediction
   |
   v
Alerts & Recommendations
   |
   v
Decision Support
```

---

## Learning Outcomes

This project demonstrates practical experience with:

- Python programming
- Real-time API integration
- Data collection
- Data cleaning and preprocessing
- Exploratory Data Analysis
- Feature engineering
- Statistical analysis
- Machine learning
- Model evaluation
- Real-time data pipelines
- Interactive dashboards
- Data visualization
- Git and GitHub
- Project documentation

---

## Author

**Mohan**

Data Analysis | Data Science | Machine Learning

---

## Disclaimer

This project is developed for **educational, portfolio, and demonstration purposes**.

The City Stress Score and machine-learning predictions are experimental outputs based on project-defined methodology and available data. They should not be treated as official measurements or forecasts for public safety, health, transportation, or city management.

---

## License

This project is currently maintained as a portfolio and educational project.
