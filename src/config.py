#config.py
import os
import random
import numpy as np
import pandas as pd
import torch

DATASET_ROOT = r"C:\Users\Acer\Desktop\minor project\SRG-system\esc-50\audio\audio\44100"
METADATA_PATH = r"C:\Users\Acer\Desktop\minor project\SRG-system\esc-50\esc50.csv"
MODEL_SAVE_PATH = os.path.join("models", "home_environment_sound.pth")
LABEL_SAVE_PATH = os.path.join("models", "labels.npy")
BASE_DIR = r"C:\Users\Acer\Desktop\minor project\SRG-system"
MFCC_PATH = os.path.join(BASE_DIR,"data_folder","MFCCS.npy")
NP_LABELS_PATH = os.path.join(BASE_DIR,"data_folder","LABELS.npy")
TRAIN_MEAN_PATH = os.path.join(BASE_DIR, "data_folder", "train_mean.npy")
TRAIN_STD_PATH = os.path.join(BASE_DIR, "data_folder", "train_std.npy")

TARGET_CLASSES = [
    'crackling_fire',
    'rain',
    'crying_baby',
    'door_wood_knock',
    'clapping',
    'glass_breaking',
    'siren',
    'clock_alarm',
    'dog',
    'cat'
]
NUM_CLASSES = len(TARGET_CLASSES)
dataset = pd.read_csv(METADATA_PATH)
dataset = dataset[dataset['category'].isin(TARGET_CLASSES)]
all_files = [os.path.join(DATASET_ROOT, f) for f in dataset['filename']]

SEED = 42
def seed_everything(): 
    random.seed(SEED) 
    os.environ['PYTHONHASHSEED'] = str(SEED) 
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

SAMPLE_RATE = 16000
DURATION = 3.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)

N_FFT = 512
HOP_LENGTH = 256

BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.001406
N_MFCCS = 40

DROPOUT_RATE = 0.4638
BASE_FILTERS = 8
WEIGHT_DECAY = 7.5e-5

DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

decoder = {0: 'crackling_fire', 
           1: 'rain', 
           2: 'crying_baby', 
           3: 'door_wood_knock', 
           4: 'clapping', 
           5: 'glass_breaking', 
           6: 'siren', 
           7: 'clock_alarm', 
           8: 'dog', 
           9: 'cat'}

encoder = {'crackling_fire': 0,
    'rain': 1,
    'crying_baby': 2,
    'door_wood_knock': 3,
    'clapping': 4,
    'glass_breaking': 5,
    'siren': 6,
    'clock_alarm': 7,
    'dog': 8,
    'cat': 9}
