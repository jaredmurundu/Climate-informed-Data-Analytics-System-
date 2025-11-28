"""
=============================================================
 INTEGRATED CLIMATE–HEALTH DECISION SUPPORT SYSTEM (FINAL)
 Author: Jared Murundu
 Purpose: Predict malaria risk using integrated climate-health data
=============================================================
"""

import pandas as pd
import numpy as np
import datetime
import pyodbc
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn.neural_network import MLPRegressor
from fastapi import FastAPI
import uvicorn
import os


# ---------------------------------------------------------
# DATABASE CONNECTION
# ---------------------------------------------------------
def get_connection():
    return pyodbc.connect(
        "Driver={ODBC Driver 17 for SQL Server};"
        "Server=DESKTOP-4OAA696\\MSSQLSERVER01;"
        "Database=ClimateHealthSystem;"
        "Trusted_Connection=yes;"
    )


# ---------------------------------------------------------
# LOAD INTEGRATED CLIMATE + HEALTH EXCEL INTO SQL
# ---------------------------------------------------------
def load_integrated_excel():
    df = pd.read_excel("climate_data_2015_to_2024.xlsx")

    df = df.rename(columns={
        "Month": "date",
        "Temperature (°C)": "temperature",
        "Rainfall (mm)": "rainfall",
        "Humidity (%)": "humidity",
        "Wind Speed (m/s)": "wind_speed",
        "Elevation (m)": "elevation",
        "Wind Pattern Index": "wind_pattern",
        "Malaria Cases": "malaria_cases",
        "Facility ID": "facility_id"
    })

    print("Uploading climate-health data to SQL...")

    conn = get_connection()
    cursor = conn.cursor()

    for _, row in df.iterrows():
        cursor.execute("""
            INSERT INTO climate_health_data 
            (date, temperature, rainfall, humidity, wind_speed, elevation, wind_pattern, malaria_cases, facility_id)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        row["date"], row["temperature"], row["rainfall"], row["humidity"],
        row["wind_speed"], row["elevation"], row["wind_pattern"],
        row["malaria_cases"], row.get("facility_id", None)
        )

    conn.commit()
    cursor.close()
    conn.close()

    print("✔ Climate-health data uploaded successfully.")


# ---------------------------------------------------------
# TRAIN ENSEMBLE MODEL FROM SQL DATA
# ---------------------------------------------------------
def train_model():
    print("Training model...")

    # Load data directly from SQL
    conn = get_connection()
    df = pd.read_sql("SELECT * FROM climate_health_data", conn)
    conn.close()

    X = df[['temperature', 'rainfall', 'humidity',
            'wind_speed', 'elevation', 'wind_pattern']]
    y = df['malaria_cases']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, shuffle=False
    )

    # Random Forest
    rf = RandomForestRegressor(n_estimators=200)
    rf.fit(X_train, y_train)

    # XGBoost
    xgb = XGBRegressor(n_estimators=200, learning_rate=0.1)
    xgb.fit(X_train, y_train)

    # ANN (MLPRegressor)
    ann = MLPRegressor(hidden_layer_sizes=(32, 16),
                       activation="relu",
                       max_iter=500)
    ann.fit(X_train, y_train)

    # Ensemble predictions
    ensemble_pred = (rf.predict(X_test) +
                     xgb.predict(X_test) +
                     ann.predict(X_test)) / 3

    print("R2 Score:", r2_score(y_test, ensemble_pred))
    print("MAE:", mean_absolute_error(y_test, ensemble_pred))

    # Save models
    joblib.dump(rf, "rf_model.pkl")
    joblib.dump(xgb, "xgb_model.pkl")
    joblib.dump(ann, "ann_model.pkl")

    print("✔ Models saved successfully.")


# ---------------------------------------------------------
# PREDICT AND SAVE TO SQL
# ---------------------------------------------------------
def get_risk_level(value):
    if value >= 0.7:
        return "RED"
    elif value >= 0.4:
        return "AMBER"
    return "GREEN"


def predict_and_save(temp, rain, humid, wind, elev, pattern):
    if not os.path.exists("rf_model.pkl"):
        raise FileNotFoundError("Models not trained yet. Run train_model() first.")

    rf = joblib.load("rf_model.pkl")
    xgb = joblib.load("xgb_model.pkl")
    ann = joblib.load("ann_model.pkl")

    X = pd.DataFrame([[temp, rain, humid, wind, elev, pattern]],
                     columns=['temperature', 'rainfall', 'humidity',
                              'wind_speed', 'elevation', 'wind_pattern'])

    final = (rf.predict(X)[0] +
             xgb.predict(X)[0] +
             ann.predict(X)[0]) / 3

    risk = get_risk_level(final)

    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("EXEC InsertPrediction ?, ?, ?",
                   datetime.date.today(),
                   float(final),
                   risk)
    conn.commit()
    cursor.close()
    conn.close()

    return final, risk


# ---------------------------------------------------------
# FASTAPI ENDPOINT
# ---------------------------------------------------------
app = FastAPI()

@app.get("/predict")
def predict(
    temperature: float,
    rainfall: float,
    humidity: float,
    wind_speed: float,
    elevation: float,
    wind_pattern: float
):
    pred, risk = predict_and_save(
        temperature, rainfall, humidity,
        wind_speed, elevation, wind_pattern
    )

    return {
        "predicted_cases": pred,
        "risk_level": risk,
        "message": "Prediction saved successfully."
    }


# ---------------------------------------------------------
# RUN SYSTEM (TRAIN FIRST TIME ONLY)
# ---------------------------------------------------------
if __name__ == "__main__":
    # First run ONLY:
    # load_integrated_excel()
    # train_model()

    uvicorn.run(app, host="0.0.0.0", port=8000)
