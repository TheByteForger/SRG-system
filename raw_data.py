#raw_data.py
import sounddevice as sd
import numpy as np
import torch
import librosa
from src.config import SAMPLE_RATE, NUM_SAMPLES, N_MFCCS, N_FFT, HOP_LENGTH, TRAIN_MEAN_PATH, TRAIN_STD_PATH

def record_mic_for_model(duration=3.0):
    print(f"Recording {duration} seconds...")
    recording = sd.rec(int(duration * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1)
    sd.wait()
    recording = recording.flatten()

    if len(recording) < NUM_SAMPLES:
        pad_width = NUM_SAMPLES - len(recording)
        recording = np.pad(recording, (0, pad_width))
    elif len(recording) > NUM_SAMPLES:
        recording = recording[:NUM_SAMPLES]

    recording = recording / np.max(np.abs(recording))

    # 4️⃣ Compute MFCCs
    mfccs = librosa.feature.mfcc(
        y=recording,
        sr=SAMPLE_RATE,
        n_mfcc=N_MFCCS,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH
    )

    # 5️⃣ Load training mean and std
    train_mean = np.load(TRAIN_MEAN_PATH).squeeze()  # shape becomes (40,)
    train_std = np.load(TRAIN_STD_PATH).squeeze()    # shape becomes (40,)

    # 6️⃣ Normalize MFCCs same as training
    mfccs = (mfccs - train_mean) / train_std

    # 7️⃣ Convert to tensor
    mfcc_tensor = torch.tensor(mfccs, dtype=torch.float32).unsqueeze(0)  # shape: (1, 40, time_frames)

    return mfcc_tensor

# Example usage:
mfcc_input = record_mic_for_model()
print(mfcc_input.shape)
