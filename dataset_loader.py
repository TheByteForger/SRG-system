import numpy as np
import torch
import src.config as config
from torch.utils.data import Dataset, DataLoader
import os

class MFCCDataset(Dataset):
    def __init__(self, mfcc_file="MFCCS.npy", labels_file="LABELS.npy"):
        if not os.path.exists(mfcc_file) or not os.path.exists(labels_file):
            raise FileNotFoundError(
                f"{mfcc_file} or {labels_file} not found. Please run dataset.py first."
            )
        self.X = np.load(mfcc_file)
        self.y = np.load(labels_file)
        
        self.X = torch.tensor(self.X, dtype=torch.float32)
        self.y = torch.tensor(self.y, dtype=torch.long)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


def get_dataloader(batch_size=32, shuffle=True, mfcc_file="MFCCS.npy", labels_file="LABELS.npy"):
    
    dataset = MFCCDataset(mfcc_file, labels_file)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return dataloader