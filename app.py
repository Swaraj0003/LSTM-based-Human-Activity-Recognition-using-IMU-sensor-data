import streamlit as st
import pandas as pd
import numpy as np
from src.predict import load_artifacts, create_sequences_predict, ACTIVITY_MAP

st.set_page_config(page_title="HAR System", layout="centered")

st.title("🏃 Human Activity Recognition System")
st.write("IMU-based Activity Detection using LSTM")

model, scaler = load_artifacts()

uploaded_file = st.file_uploader(
    "Upload IMU CSV File (ax, ay, az, gx, gy, gz only)",
    type=["csv"]
)

if uploaded_file:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Data Preview")
    st.dataframe(df.head())

    # Validate columns
    required_cols = ["ax", "ay", "az", "gx", "gy", "gz"]
    if not all(col in df.columns for col in required_cols):
        st.error("CSV must contain ax, ay, az, gx, gy, gz columns")
    else:
        X = df[required_cols]
        X_scaled = scaler.transform(X)
        X_seq = create_sequences_predict(X_scaled)

        preds = model.predict(X_seq)
        predicted_class = np.argmax(preds[-1])
        activity = ACTIVITY_MAP[predicted_class]

        st.subheader("🧠 Predicted Activity")
        st.success(activity)

        st.subheader("📊 Accelerometer Signals")
        st.line_chart(df[["ax", "ay", "az"]])
