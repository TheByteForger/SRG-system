import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
import src.config as config

class MFCCDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloaders(batch_size=config.BATCH_SIZE, mfcc_file="MFCCS.npy", labels_file="LABELS.npy"):
    if not os.path.exists(mfcc_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"{mfcc_file} or {labels_file} not found. Please run dataset.py first.")

    X = np.load(mfcc_file)
    y = np.load(labels_file)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=y
    )


    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.SEED, stratify=y_temp
    )


    train_dataset = MFCCDataset(X_train, y_train)
    val_dataset   = MFCCDataset(X_val, y_val)
    test_dataset  = MFCCDataset(X_test, y_test)


    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
