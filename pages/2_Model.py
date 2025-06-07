import streamlit as st
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix
import joblib
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor

st.title("🤖 Model Pelatihan")

# Load data
df = pd.read_csv("data/semarang_resto_dataset.csv")

# Preprocessing
df = df.dropna()
if 'resto_rating' not in df.columns:
    st.error("Kolom 'resto_rating' tidak ditemukan.")
    st.stop()

X = df.drop('resto_rating', axis=1).select_dtypes(include='number')
y = df['resto_rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Model training
model = RandomForestRegressor()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# Save model
joblib.dump(model, "model/trained_model.pkl")

st.subheader("Classification Report")
st.text(classification_report(y_test, y_pred))

st.subheader("Confusion Matrix")
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
st.pyplot(plt)
