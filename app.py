import streamlit as st
import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Heart Disease DL Dashboard", layout="wide")

@st.cache_resource
def load_artifacts():
    model = tf.keras.models.load_model("models/dl_model.keras")
    scaler = joblib.load("models/scaler.pkl")
    feature_cols = joblib.load("models/feature_cols.pkl")
    return model, scaler, feature_cols

@st.cache_data
def load_data():
    return pd.read_csv("data/heart.csv")

def risk_label(p):
    if p < 0.30: return "Low"
    if p < 0.60: return "Medium"
    if p < 0.80: return "High"
    return "Very High"

model, scaler, feature_cols = load_artifacts()
df = load_data()

st.title("Heart Disease Prediction (Deep Learning)")

tab1, tab2 = st.tabs(["Prediction", "Dashboard"])

with tab1:
    st.subheader("Enter patient values")

    cols = st.columns(3)

    # Inputs (match the features used in training)
    with cols[0]:
        age = st.number_input("age", 1, 120, 50)
        sex = st.selectbox("sex (1=male,0=female)", [0, 1])
        cp = st.selectbox("cp", [0, 1, 2, 3])
        trestbps = st.number_input("trestbps", 50, 250, 120)
        chol = st.number_input("chol", 100, 700, 200)

    with cols[1]:
        fbs = st.selectbox("fbs", [0, 1])
        restecg = st.selectbox("restecg", [0, 1, 2])
        thalach = st.number_input("thalach", 50, 250, 150)
        exang = st.selectbox("exang", [0, 1])
        oldpeak = st.number_input("oldpeak", 0.0, 10.0, 1.0, step=0.1)

    with cols[2]:
        slope = st.selectbox("slope", [0, 1, 2])
        ca = st.selectbox("ca", [0, 1, 2, 3])
        thal = st.selectbox("thal", [0, 1, 2, 3])

    input_dict = {
        "age": age, "sex": sex, "cp": cp, "trestbps": trestbps, "chol": chol,
        "fbs": fbs, "restecg": restecg, "thalach": thalach, "exang": exang,
        "oldpeak": oldpeak, "slope": slope, "ca": ca, "thal": thal
    }

    input_df = pd.DataFrame([input_dict], columns=feature_cols)
    X_scaled = scaler.transform(input_df)

    if st.button("Predict"):
        prob = float(model.predict(X_scaled)[0][0])
        pred = 1 if prob >= 0.5 else 0

        if pred == 1:
            st.error(f"Prediction: Heart disease likely | Risk={prob*100:.1f}% | Level={risk_label(prob)}")
        else:
            st.success(f"Prediction: No heart disease likely | Risk={prob*100:.1f}% | Level={risk_label(prob)}")

        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=prob * 100,
            title={"text": "Risk (%)"},
            gauge={"axis": {"range": [0, 100]},
                   "steps": [
                       {"range": [0, 30], "color": "lightgreen"},
                       {"range": [30, 60], "color": "khaki"},
                       {"range": [60, 80], "color": "orange"},
                       {"range": [80, 100], "color": "red"},
                   ]}
        ))
        st.plotly_chart(fig, use_container_width=True)

with tab2:
    st.subheader("Dataset dashboard")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Records", len(df))
    c2.metric("Disease=1", int(df["target"].sum()))
    c3.metric("Disease=0", int((df["target"] == 0).sum()))
    c4.metric("Avg age", f"{df['age'].mean():.1f}")

    colA, colB = st.columns(2)

    with colA:
        fig = px.pie(df, names="target", title="Target distribution")
        st.plotly_chart(fig, use_container_width=True)

    with colB:
        fig = px.histogram(df, x="age", color="target", barmode="overlay", title="Age vs target")
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Correlation heatmap")
    corr = df.corr(numeric_only=True)
    fig = px.imshow(corr, aspect="auto", title="Correlation matrix")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Data preview")
    st.dataframe(df.head(30), use_container_width=True)