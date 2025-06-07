import streamlit as st
import pandas as pd
import joblib

st.title("🔮 Prediksi Nilai Resto Rating")

# Load model
try:
    model = joblib.load("model/trained_model.pkl")
except:
    st.error("Model belum tersedia. Silakan latih model terlebih dahulu.")
    st.stop()

# Ambil nama fitur dari model
feature_names = model.feature_names_in_

st.subheader("Isi Formulir Data Restoran")
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", step=1.0)

# Prediksi
if st.button("Prediksi"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Prediksi rating restoran: **{prediction:.2f}**")
