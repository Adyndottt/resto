import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

st.title("📊 Exploratory Data Analysis (EDA)")

# Load data
df = pd.read_csv("data/semarang_resto_dataset.csv")

st.subheader("Preview Data")
st.dataframe(df.head())

st.subheader("Informasi Dataset")
st.write(df.describe())
st.write("Shape:", df.shape)

st.subheader("Distribusi Kolom resto_rating")
if 'resto_rating' in df.columns:
    st.bar_chart(df['resto_rating'].value_counts())
else:
    st.warning("Kolom 'resto_rating' tidak ditemukan dalam dataset.")

st.subheader("Heatmap Korelasi")
numeric_cols = df.select_dtypes(include='number')
plt.figure(figsize=(10, 6))
sns.heatmap(numeric_cols.corr(), annot=True, cmap='coolwarm')
st.pyplot(plt)
