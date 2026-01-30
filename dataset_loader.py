#dataset_loader.py
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import os
from sklearn.model_selection import train_test_split
import src.config as config

config.seed_everything()

class MFCCDataset(Dataset):
    def __init__(self, X, y, augment=False, freq_mask_param = 5, time_mask_param = 20, noise_level = 0.01, shift_max = 10):
        self.X = torch.from_numpy(X).float()
        self.y = torch.from_numpy(y).long()
        self.augment = augment
        self.freq_mask_param = freq_mask_param
        self.time_mask_param = time_mask_param
        self.noise_level = noise_level
        self.shift_max = shift_max

        
    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        mfcc = self.X[idx].clone()
        label = self.y[idx]

        if self.augment:
            mfcc = self.spec_augment(mfcc)
        return mfcc, label
    
    def spec_augment(self, mfcc):
        n_mfcc = mfcc.shape[1]
        n_frames = mfcc.shape[2]

        f = np.random.randint(0, self.freq_mask_param + 1)
        f0 = np.random.randint(0, max(1, n_mfcc - f))
        mfcc[:, f0:f0+f, :] = 0

        t = np.random.randint(0, self.time_mask_param + 1)
        t0 = np.random.randint(0, max(1, n_frames - t)) 
        mfcc[:, :, t0:t0+t] = 0

        mfcc = self.time_shift(mfcc, self.shift_max)

        mfcc = self.add_noise(mfcc, self.noise_level)

        return mfcc
    
    def time_shift(self, mfcc, shift_max):
        n_frames = mfcc.shape[2]
        shift = np.random.randint(-shift_max, shift_max + 1)
        if shift == 0:
            return mfcc

        device = mfcc.device

        if shift > 0:
            mfcc = torch.cat([
                mfcc[:, :, shift:], 
                torch.zeros((mfcc.shape[0], mfcc.shape[1], shift), device=device)
                ], dim=2)
        else:
            shift = -shift
            mfcc = torch.cat([
            torch.zeros((mfcc.shape[0], mfcc.shape[1], shift), device=device), 
            mfcc[:, :, :-shift]
                ], dim=2)

        return mfcc

    
    def add_noise(self, mfcc, noise_level):
        noise = torch.randn_like(mfcc) * noise_level
        return mfcc + noise


def get_dataloaders(batch_size=config.BATCH_SIZE, mfcc_file=config.MFCC_PATH, labels_file=config.NP_LABELS_PATH):
    if not os.path.exists(mfcc_file) or not os.path.exists(labels_file):
        raise FileNotFoundError(f"{mfcc_file} or {labels_file} not found. Please run dataset.py first.")

    X = np.load(mfcc_file)
    y = np.load(labels_file)

    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.2, random_state=config.SEED, stratify=y
    )

    train_mean = X_train.mean(axis = (0, 2), keepdims = True)
    train_std = X_train.std(axis=(0, 2), keepdims = True) + 1e-6

    np.save(config.TRAIN_MEAN_PATH, train_mean)
    np.save(config.TRAIN_STD_PATH, train_std)

    X_train = (X_train - train_mean) / train_std

    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=config.SEED, stratify=y_temp
    )
    X_val = (X_val - train_mean) / train_std
    X_test = (X_test - train_mean) / train_std

    train_dataset = MFCCDataset(X_train, y_train, augment= True)
    val_dataset   = MFCCDataset(X_val, y_val, augment= False)
    test_dataset  = MFCCDataset(X_train, y_train, augment= False)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_loader  = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, test_loader
