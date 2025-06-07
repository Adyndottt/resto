import streamlit as st
import pandas as pd
import joblib

st.title("🔮 Prediksi Kategori Restoran")

# Load model
try:
    model = joblib.load("model/trained_model.pkl")
except:
    st.error("Model belum dilatih. Silakan latih di halaman Model.")
    st.stop()

# Ambil fitur yang digunakan model
feature_names = model.feature_names_in_

st.subheader("Isi Formulir")
user_input = {}
for feature in feature_names:
    user_input[feature] = st.number_input(f"{feature}", step=1.0)

# Prediksi
if st.button("Prediksi"):
    input_df = pd.DataFrame([user_input])
    prediction = model.predict(input_df)[0]
    st.success(f"Kategori restoran diprediksi sebagai: **{prediction}**")
