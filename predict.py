import torch
import src.config as config
from src.model import HomeAudioCNN
import numpy as np

# ------------------------
# Device
# ------------------------
device = config.DEVICE

# ------------------------
# Label Decoder
# ------------------------
decoder = {
    0: 'crackling_fire',
    1: 'rain',
    2: 'crying_baby',
    3: 'door_wood_knock',
    4: 'door_wood_creaks',
    5: 'glass_breaking',
    6: 'siren',
    7: 'clock_alarm',
    8: 'dog',
    9: 'cat'
}

# ------------------------
# Load Model
# ------------------------
model = HomeAudioCNN(num_classes=config.NUM_CLASSES)
model.load_state_dict(
    torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only=True)
)
model.to(device)
model.eval()

print("✅ Model loaded successfully")

# ------------------------
# INPUT MFCC VECTOR (40,)
# ------------------------
mfcc_vector = np.array([
    318.8390,7.2824,9.7342,4.2996,3.9909,5.7994,9.3523,2.8527,-1.2093,-1.5791,-3.1065,2.6236,-0.2646,-4.9743,-2.7338,1.3852,-2.7003,-2.3688,-3.6058,-0.2727,1.9096,-2.3214,-3.6865,-1.5391,2.2374,4.8752,0.7945,0.2484,-0.2018,-1.0178,2.3142,2.9675,1.1170,-0.9276,-0.2271,-1.9401,-1.2238,0.0410,-1.4291,-0.6125
], dtype=np.float32)

# ------------------------
# Expand to (40, 216)
# ------------------------
TARGET_FRAMES = 216

mfcc_map = np.tile(mfcc_vector[:, np.newaxis], (1, TARGET_FRAMES))

# Shape: (1, 1, 40, 216)
mfcc_tensor = torch.tensor(mfcc_map).unsqueeze(0).unsqueeze(0)
mfcc_tensor = mfcc_tensor.to(device)

# ------------------------
# Prediction
# ------------------------
with torch.no_grad():
    logits = model(mfcc_tensor)
    probs = torch.softmax(logits, dim=1)
    pred_idx = torch.argmax(probs, dim=1).item()
    confidence = probs[0][pred_idx].item()

# ------------------------
# Output
# ------------------------
print("\n🎯 Prediction Result")
print("--------------------")
print(f"Predicted Class : {decoder[pred_idx]}")
print(f"Confidence      : {confidence:.4f}")

print("\n📊 Class Probabilities:")
for i, p in enumerate(probs[0]):
    print(f"{decoder[i]:20s}: {p.item():.4f}")
