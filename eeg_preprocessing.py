import mne
from scipy.io import loadmat
import numpy as np

REQUIRED_CHANNELS = [
    'Fp1','Fp2','F3','F4','C3','C4','P3','P4','O1','O2',
    'F7','F8','T7','T8','P7','P8','Fz','Cz','Pz'
]

def load_eeg(uploaded_file, file_type):
    if file_type == "MAT":
        mat = loadmat(uploaded_file)
        key = [k for k in mat.keys() if not k.startswith("__")][0]
        eeg = mat[key]

    else:  # RAW EEG (EDF)
        raw = mne.io.read_raw_edf(uploaded_file, preload=True)
        raw.pick_channels(REQUIRED_CHANNELS)
        raw.resample(128)
        eeg = raw.get_data().T

    return eeg
