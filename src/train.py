import numpy as np
from sklearn.model_selection import train_test_split
from data_loader import load_data
from preprocessing import scale_features, create_sequences, normalize_labels
from model_builder import build_lstm

# Load dataset
df = load_data(r"C:\Users\USER\Desktop\swaraj_data_science_project\HAR_streamlit_app\data\IMU-based Human Activity Recignition Dataset.csv")

# Features & labels
y_raw = df["activity"].values
y = normalize_labels(y_raw)  # 1–6 → 0–5

X_scaled = scale_features(df, "activity")

# Create sequences
X, y = create_sequences(X_scaled, y, time_steps=50)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# Build model
model = build_lstm(
    input_shape=(X.shape[1], X.shape[2]),
    num_classes=6
)

# Train
model.fit(
    X_train,
    y_train,
    epochs=20,
    batch_size=32,
    validation_split=0.2
)

# Save model
model.save("model/lstm_model.h5")
print("✅ Model trained successfully")
