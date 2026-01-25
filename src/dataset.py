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

mfcc = []
for file in all_files:
    audio, sr = librosa.load(file, sr=config.SAMPLE_RATE)
    mfccs = compute_mfcc_from_audio(all_files)
    mfcc.append(mfcc)
mfcc = np.array(mfcc)
normalized_mfcc = normalize_mfcc(mfccs)