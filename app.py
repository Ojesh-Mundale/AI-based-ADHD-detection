import streamlit as st
import numpy as np
import pandas as pd
from scipy.io import loadmat
import tensorflow as tf
import joblib
import mne
import tempfile
import os

# =============================
# Load trained model & scaler
# =============================
model = tf.keras.models.load_model("model.h5")
scaler = joblib.load("scaler.pkl")

# =============================
# UI
# =============================
st.title("EEG-based ADHD Detection System")
st.write("Upload EEG data to predict ADHD (MAT / EDF / CSV / TXT)")

# =============================
# Required EEG channels (19)
# =============================
REQUIRED_CHANNELS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz'
]

# =============================
# File selection
# =============================
file_type = st.selectbox(
    "Select EEG file type",
    ["MAT", "EDF", "CSV / TXT"]
)

uploaded_file = st.file_uploader(
    "Upload EEG file",
    type=["mat", "edf", "csv", "txt"]
)

# =============================
# Windowing parameters
# =============================
WINDOW_SIZE = 256
STEP_SIZE = 128

# =============================
# Helper functions
# =============================
def create_windows(eeg):
    """eeg shape: (channels, time)"""
    windows = []
    for i in range(0, eeg.shape[1] - WINDOW_SIZE, STEP_SIZE):
        windows.append(eeg[:, i:i + WINDOW_SIZE])
    return np.array(windows)

def load_mat_file(file):
    mat = loadmat(file)
    key = [k for k in mat.keys() if not k.startswith("__")][0]
    eeg = mat[key].T
    return eeg

def load_csv_or_txt(file):
    try:
        df = pd.read_csv(file)
        df = df.select_dtypes(include=[np.number])
        eeg = df.values.T
        return eeg
    except Exception:
        file.seek(0)
        data = np.loadtxt(file)
        eeg = data.T
        return eeg

# =============================
# Main logic
# =============================
if uploaded_file is not None:
    try:
        # -------- Load EEG --------
        if file_type == "MAT":
            eeg = load_mat_file(uploaded_file)

        elif file_type == "EDF":
            # Save EDF temporarily (Streamlit fix)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".edf") as tmp_file:
                tmp_file.write(uploaded_file.read())
                temp_path = tmp_file.name

            raw = mne.io.read_raw_edf(temp_path, preload=True, verbose=False)
            os.remove(temp_path)

            missing = set(REQUIRED_CHANNELS) - set(raw.ch_names)
            if missing:
                st.error(f"Missing required channels: {missing}")
                st.stop()

            raw.pick_channels(REQUIRED_CHANNELS)
            raw.resample(128)
            eeg = raw.get_data()

        else:  # CSV / TXT
            eeg = load_csv_or_txt(uploaded_file)

        # -------- Validation --------
        if eeg.shape[0] != 19:
            st.error("EEG must have exactly 19 channels")
            st.stop()

        if eeg.shape[1] < WINDOW_SIZE:
            st.error("EEG recording too short for prediction")
            st.stop()

        st.success("EEG loaded successfully")
        st.write("EEG shape (channels × time):", eeg.shape)

        # -------- Prediction --------
        if st.button("Predict ADHD"):
            windows = create_windows(eeg)

            reshaped = windows.reshape(-1, windows.shape[-1])
            scaled = scaler.transform(reshaped)
            scaled = scaled.reshape(windows.shape)

            X = scaled[..., np.newaxis]

            preds = model.predict(X, verbose=0)
            mean_pred = float(np.mean(preds))

            if mean_pred < 0.5:
                st.error("ADHD Detected")
            else:
                st.success("No ADHD Detected")

            confidence = abs(mean_pred - 0.5) * 2 * 100
            st.write("Prediction confidence:", round(confidence, 2), "%")

    except Exception as e:
        st.error(f"Error: {str(e)}")
