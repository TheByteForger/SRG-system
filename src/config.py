import os
import random
import numpy as np
import pandas as pd
import torch

DATASET_ROOT = r"C:\Users\Acer\Desktop\SRG\esc-50\audio\audio\44100"
METADATA_PATH = r"C:\Users\Acer\Desktop\SRG\esc-50\esc50.csv"
MODEL_SAVE_PATH = os.path.join("models", "home_environment_sound.pth")
LABEL_SAVE_PATH = os.path.join("models", "labels.npy")
BASE_DIR = r"C:\Users\Acer\Desktop\SRG"
MFCC_PATH = os.path.join(BASE_DIR, "MFCCS.npy")
NP_LABELS_PATH = os.path.join(BASE_DIR, "LABELS.npy")

TARGET_CLASSES = [
    'crackling_fire',
    'rain',
    'crying_baby',
    'door_wood_knock',
    'door_wood_creaks',
    'glass_breaking',
    'siren',
    'clock_alarm',
    'dog',
    'cat'
]

dataset = pd.read_csv(METADATA_PATH)
dataset = dataset[dataset['category'].isin(TARGET_CLASSES)]
all_files = [os.path.join(DATASET_ROOT, f) for f in dataset['filename']]

SEED = 42
def seed_everything(): 
    random.seed(SEED) 
    os.environ['PYTHONHASHSEED'] = str(SEED) 
    np.random.seed(SEED) 

SAMPLE_RATE = 16000
DURATION = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

N_FFT = 512
HOP_LENGTH = 256
N_MELS = 13

BATCH_SIZE = 16
EPOCHS = 20
LEARNING_RATE = 0.001
N_MFCCS = 13

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

decoder = {0: 'crackling_fire', 
           1: 'rain', 
           2: 'crying_baby', 
           3: 'door_wood_knock', 
           4: 'door_wood_creaks', 
           5: 'glass_breaking', 
           6: 'siren', 
           7: 'clock_alarm', 
           8: 'dog', 
           9: 'cat'}

encoder = {'crackling_fire': 0,
    'rain': 1,
    'crying_baby': 2,
    'door_wood_knock': 3,
    'door_wood_creaks': 4,
    'glass_breaking': 5,
    'siren': 6,
    'clock_alarm': 7,
    'dog': 8,
    'cat': 9}

