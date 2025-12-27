import numpy as np
from sklearn.preprocessing import MinMaxScaler
import pickle

def scale_features(df, label_column):
    scaler = MinMaxScaler()
    X = df.drop(label_column, axis=1)
    X_scaled = scaler.fit_transform(X)

    with open("model/scaler.pkl", "wb") as f:
        pickle.dump(scaler, f)

    return X_scaled

def normalize_labels(y):
    """
    Convert labels from 1–6 → 0–5
    """
    return y - 1

def create_sequences(X, y, time_steps=50):
    X_seq, y_seq = [], []
    for i in range(len(X) - time_steps):
        X_seq.append(X[i:i + time_steps])
        y_seq.append(y[i + time_steps])
    return np.array(X_seq), np.array(y_seq)
