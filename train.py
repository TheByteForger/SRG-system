import os
import torch
import src.config as config
from dataset_loader import get_dataloader

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = config.DEVICE
loader = get_dataloader(batch_size=config.BATCH_SIZE)

for batch_X, batch_y in loader:
    batch_X, batch_y = batch_X.to(device), batch_y.to(device)
    print(batch_X.shape)
    print(batch_y.shape)
    break
