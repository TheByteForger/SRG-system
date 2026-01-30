#test.py
import torch
import src.config as config
from src.model import HomeAudioCNN
from dataset_loader import get_dataloaders

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
import seaborn as sns

# ------------------------
# Device
# ------------------------
device = config.DEVICE

# ------------------------
# Load model
# ------------------------
model = HomeAudioCNN(num_classes=config.NUM_CLASSES)
model.load_state_dict(
    torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True)
)
model.to(device)
model.eval()

# ------------------------
# Load test data
# ------------------------
_, _, test_loader = get_dataloaders(batch_size=config.BATCH_SIZE)

# ------------------------
# Test + collect predictions
# ------------------------
correct = 0
total = 0

all_preds = []
all_labels = []

with torch.no_grad():
    for batch_X, batch_y in test_loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_X)
        _, preds = torch.max(outputs, 1)

        correct += (preds == batch_y).sum().item()
        total += batch_y.size(0)

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(batch_y.cpu().numpy())

test_acc = correct / total
print(f"Test Accuracy: {test_acc:.4f}")

# ------------------------
# Confusion Matrix
# ------------------------
cm = confusion_matrix(all_labels, all_preds)

# ------------------------
# Plot
# ------------------------
plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=config.encoder.keys(),
    yticklabels=config.encoder.keys()
)

plt.xlabel("Predicted Label")
plt.ylabel("True Label")
plt.title("Confusion Matrix (Test Set)")
plt.tight_layout()
plt.show()