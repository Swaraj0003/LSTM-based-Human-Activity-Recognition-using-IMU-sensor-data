import numpy as np
import pickle
from tensorflow.keras.models import load_model

ACTIVITY_MAP = {
    0: "Walking Upstairs",
    1: "Walking Downstairs",
    2: "Walking",
    3: "Sitting",
    4: "Standing",
    5: "Jogging"
}

def load_artifacts():
    model = load_model("model/lstm_model.h5")
    scaler = pickle.load(open("model/scaler.pkl", "rb"))
    return model, scaler

def create_sequences_predict(X, time_steps=50):
    sequences = []
    for i in range(len(X) - time_steps):
        sequences.append(X[i:i + time_steps])
    return np.array(sequences)
