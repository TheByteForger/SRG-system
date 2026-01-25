import numpy as np
import pandas as pd
import librosa
import os

import config

config.seed_everything()

all_files = config.all_files

def compute_mfcc_from_audio(audio, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCCS, max_len=216):
    """
    Computes MFCC of a raw audio array and pads/truncates to fixed length.
    """
    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)
    if mfcc.shape[1] < max_len:
        mfcc = np.pad(mfcc, ((0,0),(0,max_len - mfcc.shape[1])), mode='constant')
    else:
        mfcc = mfcc[:, :max_len]
    return mfcc

def normalize_mfcc(mfcc):
    mean = mfcc.mean()
    std = mfcc.std()
    normalized_mfccs = (mfcc - mean)/std
    return normalized_mfccs


if os.path.exists("MFCCS.npy") and os.path.exists("LABELS.npy"):
    MFCCs = np.load("MFCCS.npy")
    Labels = np.load("LABELS.npy")
else:
    mfcc_list = []
    labels = []
    dataset = config.dataset
    encoded_label = config.encoder
    sr = config.SAMPLE_RATE
    for file, class_name in zip(all_files, dataset['category']):
        label = encoded_label[class_name]
        audio, _ = librosa.load(file, sr=sr)
        mfcc = compute_mfcc_from_audio(audio)
        mfcc_list.append(normalize_mfcc(mfcc))
        labels.append(label)
    mfcc_list = np.array(mfcc_list, dtype=np.float32)
    mfcc_list = mfcc_list[:, np.newaxis, :, :]
    labels = np.array(labels, dtype= np.int64)

    np.save(config.MFCC_PATH, mfcc_list)
    np.save(config.NP_LABELS_PATH, labels)


    print(f"First MFCC shape: {mfcc_list[0].shape}")
    print(f"First MFCC sample:\n{mfcc_list[0]}")
    print(f"First label: {labels[0]}")




