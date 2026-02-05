# dataset.py
import numpy as np
import pandas as pd
import librosa
import os
import config

config.seed_everything()

all_files = config.all_files

def compute_mfcc_from_audio(audio, sr=config.SAMPLE_RATE, n_mfcc=config.N_MFCCS, max_len=216):
    
    if len(audio) < sr * 0.1:
        pad_len = int(sr * 0.1) - len(audio)
        audio = np.pad(audio, (0, pad_len))

    mfcc = librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=n_mfcc)

    
    if mfcc.shape[1] < 9:
        pad_width = 9 - mfcc.shape[1]
        mfcc = np.pad(mfcc, ((0, 0), (0, pad_width)), mode='constant')
    
    
    mfcc_delta = librosa.feature.delta(mfcc)
    mfcc_delta2 = librosa.feature.delta(mfcc, order=2)


    combined = np.stack([mfcc, mfcc_delta, mfcc_delta2], axis=0)

    if combined.shape[2] < max_len:
        pad_width = max_len - combined.shape[2]
        
        combined = np.pad(combined, ((0,0), (0,0), (0, pad_width)), mode='constant')
    else:
        combined = combined[:, :, :max_len]
        
    return combined

if __name__ == "__main__":
    if os.path.exists(config.MFCC_PATH) and os.path.exists(config.NP_LABELS_PATH):
        print("Loading existing numpy files...")
        MFCCs = np.load(config.MFCC_PATH)
        Labels = np.load(config.NP_LABELS_PATH)
    else:
        print("Generating new dataset with Deltas...")
        mfcc_list = []
        labels = []
        dataset = config.dataset
        encoded_label = config.encoder
        sr = config.SAMPLE_RATE
        
        for i, (file, class_name) in enumerate(zip(all_files, dataset['category'])):
            if i % 100 == 0: 
                print(f"Processing {i}/{len(all_files)}")
            
            label = encoded_label[class_name]
            
            try:
                audio, _ = librosa.load(file, sr=sr)
                mfcc = compute_mfcc_from_audio(audio)
                mfcc_list.append(mfcc)
                labels.append(label)
            except Exception as e:
                print(f"Error processing {file}: {e}")
        mfcc_list = np.array(mfcc_list, dtype=np.float32)
        labels = np.array(labels, dtype=np.int64)

        np.save(config.MFCC_PATH, mfcc_list)
        np.save(config.NP_LABELS_PATH, labels)

        print(f"Dataset generated.")
        print(f"First MFCC shape: {mfcc_list[0].shape}")
        print(f"First label: {labels[0]}")