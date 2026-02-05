import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import src.config as config 
from src.model import HomeAudioCNN
from dataset_loader import get_dataloaders
import os

def load_trained_model(device):
    print(f"Loading model from: {config.MODEL_SAVE_PATH}")
    

    model = HomeAudioCNN(num_classes=config.NUM_CLASSES)
    
    if os.path.exists(config.MODEL_SAVE_PATH):
        state_dict = torch.load(config.MODEL_SAVE_PATH, map_location=device, weights_only= True)
        model.load_state_dict(state_dict)
        model.to(device)
        model.eval()
        return model
    else:
        raise FileNotFoundError(f"No model found at {config.MODEL_SAVE_PATH}. Train the model first!")

def evaluate_model(model, test_loader, device):
    all_preds = []
    all_labels = []
    
    print("Running evaluation on Test Set...")
    
    with torch.no_grad():
        for mfccs, labels in test_loader:
            mfccs = mfccs.to(device)
            labels = labels.to(device)
            
            outputs = model(mfccs)
            
            _, predicted = torch.max(outputs.data, 1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    return np.array(all_labels), np.array(all_preds)

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    
    plt.title('Confusion Matrix: Home Environment Sounds')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    save_path = os.path.join("models", "confusion_matrix.png")
    plt.savefig(save_path)
    print(f"Confusion Matrix saved to: {save_path}")
    plt.show()

def main():

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Testing on device: {device}")

    _, _, test_loader = get_dataloaders(batch_size=config.BATCH_SIZE)
    

    model = load_trained_model(device)
    
    y_true, y_pred = evaluate_model(model, test_loader, device)
 
    acc = accuracy_score(y_true, y_pred)
    print(f"\n{'='*30}")
    print(f"TEST ACCURACY: {acc * 100:.2f}%")
    print(f"{'='*30}\n")
    
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=config.TARGET_CLASSES))
    
    plot_confusion_matrix(y_true, y_pred, config.TARGET_CLASSES)

if __name__ == "__main__":
    main()