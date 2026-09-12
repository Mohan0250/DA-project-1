import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error


# ============================================================
# CONFIGURATION
# ============================================================

DATA_FILE = r"C:\Users\mohan\dashboard_data.csv"

FEATURES = [
    "temperature",
    "humidity",
    "wind_speed",
    "pm2_5",
    "pm10",
    "us_aqi",
    "current_speed",
    "congestion_ratio"
]

TARGET = "city_stress_score"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    df = pd.read_csv(DATA_FILE)

    df["timestamp"] = pd.to_datetime(
        df["timestamp"]
    )

    df = df.sort_values(
        "timestamp"
    ).reset_index(drop=True)

    return df


# ============================================================
# PREPARE 15-MINUTE TRAINING DATA
# ============================================================

def prepare_training_data(df):

    # Create the next timestamp
    df["next_timestamp"] = (
        df["timestamp"].shift(-1)
    )

    # Create the actual future stress
    df["future_stress"] = (
        df[TARGET].shift(-1)
    )

    # Calculate time difference
    df["time_difference"] = (
        df["next_timestamp"]
        - df["timestamp"]
    )

    # Only use records where the next observation
    # is exactly 15 minutes later
    valid = (
        (df["time_difference"] == pd.Timedelta(minutes=15))
        & df[FEATURES].notna().all(axis=1)
        & df["future_stress"].notna()
    )

    X = df.loc[
        valid,
        FEATURES
    ].copy()

    y = df.loc[
        valid,
        "future_stress"
    ].copy()

    return X, y


# ============================================================
# TRAIN MODEL
# ============================================================

def train_model():

    df = load_data()

    X, y = prepare_training_data(df)

    if len(X) < 5:

        raise ValueError(
            f"Not enough exact 15-minute training samples. "
            f"Only {len(X)} samples available."
        )

    # Time-ordered 80/20 split
    split_index = int(
        len(X) * 0.80
    )

    if split_index < 1:
        split_index = 1

    if split_index >= len(X):
        split_index = len(X) - 1

    X_train = X.iloc[
        :split_index
    ]

    X_test = X.iloc[
        split_index:
    ]

    y_train = y.iloc[
        :split_index
    ]

    y_test = y.iloc[
        split_index:
    ]

    # Random Forest model
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )

    model.fit(
        X_train,
        y_train
    )

    # Test predictions
    predictions = model.predict(
        X_test
    )

    mae = mean_absolute_error(
        y_test,
        predictions
    )

    rmse = np.sqrt(
        mean_squared_error(
            y_test,
            predictions
        )
    )

    return (
        model,
        mae,
        rmse,
        X,
        y
    )


# ============================================================
# STRESS CLASSIFICATION
# ============================================================

def classify_stress(score):

    if score <= 25:
        return "Low"

    elif score <= 50:
        return "Moderate"

    elif score <= 75:
        return "High"

    else:
        return "Critical"


# ============================================================
# NEXT 15-MINUTE LIVE PREDICTION
# ============================================================

def predict_next_stress(live_data=None):

    (
        model,
        mae,
        rmse,
        X,
        y
    ) = train_model()

    # --------------------------------------------------------
    # Use live API data
    # --------------------------------------------------------

    if live_data is not None:

        prediction_input = pd.DataFrame(
            [{
                "temperature":
                    live_data["temperature"],

                "humidity":
                    live_data["humidity"],

                "wind_speed":
                    live_data["wind_speed"],

                "pm2_5":
                    live_data["pm2_5"],

                "pm10":
                    live_data["pm10"],

                "us_aqi":
                    live_data["us_aqi"],

                "current_speed":
                    live_data["current_speed"],

                "congestion_ratio":
                    live_data["congestion_ratio"]
            }]
        )

    else:

        prediction_input = X.iloc[
            [-1]
        ]

    # Generate prediction
    predicted_score = model.predict(
        prediction_input
    )[0]

    # Keep score between 0 and 100
    predicted_score = float(
        np.clip(
            predicted_score,
            0,
            100
        )
    )

    stress_level = classify_stress(
        predicted_score
    )

    return {

        "predicted_stress":
            predicted_score,

        "stress_level":
            stress_level,

        "mae":
            mae,

        "rmse":
            rmse,

        "training_samples":
            len(X)
    }


# ============================================================
# FEATURE IMPORTANCE
# ============================================================

def get_feature_importance():

    (
        model,
        mae,
        rmse,
        X,
        y
    ) = train_model()

    importance = pd.DataFrame(
        {
            "Feature":
                FEATURES,

            "Importance":
                model.feature_importances_
        }
    )

    importance = importance.sort_values(
        "Importance",
        ascending=False
    ).reset_index(
        drop=True
    )

    return importance


# ============================================================
# PREDICTION VS ACTUAL
# ============================================================

def get_prediction_vs_actual():

    df = load_data()

    # --------------------------------------------------------
    # Create next timestamp and future actual stress
    # --------------------------------------------------------

    df["next_timestamp"] = (
        df["timestamp"].shift(-1)
    )

    df["actual_future_stress"] = (
        df[TARGET].shift(-1)
    )

    df["time_difference"] = (
        df["next_timestamp"]
        - df["timestamp"]
    )

    # --------------------------------------------------------
    # Only use exact 15-minute pairs
    # --------------------------------------------------------

    valid = (
        (df["time_difference"] == pd.Timedelta(minutes=15))
        & df[FEATURES].notna().all(axis=1)
        & df["actual_future_stress"].notna()
    )

    comparison_data = df.loc[
        valid,
        [
            "timestamp",
            "next_timestamp"
        ] + FEATURES + [
            "actual_future_stress"
        ]
    ].copy()

    if len(comparison_data) < 2:

        raise ValueError(
            "Not enough exact 15-minute pairs "
            "for Prediction vs Actual analysis."
        )

    # --------------------------------------------------------
    # Train model using the complete training set
    # --------------------------------------------------------

    X, y = prepare_training_data(
        df
    )

    model = RandomForestRegressor(
        n_estimators=100,
        random_state=42,
        max_depth=5
    )

    model.fit(
        X,
        y
    )

    # --------------------------------------------------------
    # Generate predictions
    # --------------------------------------------------------

    comparison_X = comparison_data[
        FEATURES
    ]

    predicted_values = model.predict(
        comparison_X
    )

    comparison_data[
        "predicted_future_stress"
    ] = predicted_values

    comparison_data[
        "predicted_future_stress"
    ] = comparison_data[
        "predicted_future_stress"
    ].clip(
        0,
        100
    )

    # --------------------------------------------------------
    # Calculate prediction error
    # --------------------------------------------------------

    comparison_data[
        "prediction_error"
    ] = (
        comparison_data[
            "predicted_future_stress"
        ]
        - comparison_data[
            "actual_future_stress"
        ]
    )

    comparison_data[
        "absolute_error"
    ] = comparison_data[
        "prediction_error"
    ].abs()

    return comparison_data


# ============================================================
# MAIN TEST
# ============================================================

if __name__ == "__main__":

    print()

    print(
        "=========================================="
    )

    print(
        "PREDICTIVE CITY INTELLIGENCE"
    )

    print(
        "=========================================="
    )

    try:

        # ----------------------------------------------------
        # Main prediction
        # ----------------------------------------------------

        result = predict_next_stress()

        print(
            f"Training samples: "
            f"{result['training_samples']}"
        )

        print(
            "Prediction horizon: 15 minutes"
        )

        print(
            f"Predicted City Stress: "
            f"{result['predicted_stress']:.2f}"
        )

        print(
            f"Predicted Stress Level: "
            f"{result['stress_level']}"
        )

        print(
            f"MAE: "
            f"{result['mae']:.2f}"
        )

        print(
            f"RMSE: "
            f"{result['rmse']:.2f}"
        )

        # ----------------------------------------------------
        # Feature Importance
        # ----------------------------------------------------

        print()

        print(
            "FEATURE IMPORTANCE"
        )

        print(
            "------------------------------------------"
        )

        importance = get_feature_importance()

        for _, row in importance.iterrows():

            print(
                f"{row['Feature']:<22}"
                f"{row['Importance']:.4f}"
            )

        # ----------------------------------------------------
        # Prediction vs Actual
        # ----------------------------------------------------

        print()

        print(
            "PREDICTION VS ACTUAL"
        )

        print(
            "------------------------------------------"
        )

        comparison = (
            get_prediction_vs_actual()
        )

        print(
            f"Comparison samples: "
            f"{len(comparison)}"
        )

        print()

        for _, row in comparison.iterrows():

            timestamp = row[
                "timestamp"
            ].strftime(
                "%d-%b %H:%M"
            )

            predicted = row[
                "predicted_future_stress"
            ]

            actual = row[
                "actual_future_stress"
            ]

            error = row[
                "absolute_error"
            ]

            print(
                f"{timestamp} | "
                f"Predicted: {predicted:.2f} | "
                f"Actual: {actual:.2f} | "
                f"Error: {error:.2f}"
            )

        # ----------------------------------------------------
        # Comparison metrics
        # ----------------------------------------------------

        comparison_mae = mean_absolute_error(
            comparison[
                "actual_future_stress"
            ],
            comparison[
                "predicted_future_stress"
            ]
        )

        comparison_rmse = np.sqrt(
            mean_squared_error(
                comparison[
                    "actual_future_stress"
                ],
                comparison[
                    "predicted_future_stress"
                ]
            )
        )

        print()

        print(
            "COMPARISON METRICS"
        )

        print(
            "------------------------------------------"
        )

        print(
            f"Prediction vs Actual MAE: "
            f"{comparison_mae:.2f}"
        )

        print(
            f"Prediction vs Actual RMSE: "
            f"{comparison_rmse:.2f}"
        )

        print(
            "=========================================="
        )

    except Exception as e:

        print()

        print(
            "ERROR:"
        )

        print(e)

        print(
            "=========================================="
        )