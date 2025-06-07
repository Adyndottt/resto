# pages/2_Model.py

import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import numpy as np
import joblib

st.title("📈 Pelatihan Model Regresi untuk Resto Rating")

# Load data
df = pd.read_csv("data/semarang_resto_dataset.csv")
df = df.dropna()

# Target & fitur
y = df['resto_rating']
X = df.select_dtypes(include='number').drop('resto_rating', axis=1)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Latih model regresi
model = RandomForestRegressor()
model.fit(X_train, y_train)

# Prediksi & evaluasi
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)
r2 = r2_score(y_test, y_pred)

st.subheader("📊 Evaluasi Model")
st.write(f"Mean Squared Error (MSE): {mse:.4f}")
st.write(f"Root Mean Squared Error (RMSE): {rmse:.4f}")
st.write(f"R² Score: {r2:.4f}")

# Simpan model
joblib.dump(model, "model/trained_model.pkl")
st.success("Model berhasil disimpan ke folder 'model/trained_model.pkl'")
