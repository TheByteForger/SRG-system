#train.py
import os
import torch
import torch.nn as nn
import src.config as config
from src.model import HomeAudioCNN
import torch.optim as optim
from dataset_loader import get_dataloaders

config.seed_everything()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
device = config.DEVICE
train_loader, val_loader, test_loader = get_dataloaders(batch_size=config.BATCH_SIZE)
device = config.DEVICE

batch_X, batch_y = next(iter(train_loader))
batch_X, batch_y = batch_X.to(device), batch_y.to(device)

print("Input shape: ", batch_X.shape)
print("Label shape: ", batch_y.shape)



# for batch_X, batch_y in train_loader:
#     batch_X, batch_y = batch_X.to(device), batch_y.to(device)
#     print(batch_X.shape)
#     print(batch_y.shape)
#     break

model = HomeAudioCNN(num_classes=config.NUM_CLASSES).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(
    model.parameters(),
    lr=config.LEARNING_RATE,
    weight_decay=1e-4
)

def accuracy(preds, labels):
    _, predicted = torch.max(preds, 1)
    correct = (predicted == labels).sum().item()
    total = labels.size(0)
    return correct, total


def train_one_epoch(model, loader, optimizer, criterion, device):
    model.train()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        correct, samples = accuracy(outputs, batch_y)
        total_correct += correct
        total_samples += samples

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_samples

    return avg_loss, avg_acc

@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    
    total_loss = 0.0
    total_correct = 0
    total_samples = 0
    
    for batch_X, batch_y in loader:
        batch_X = batch_X.to(device)
        batch_y = batch_y.to(device)

        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        total_loss += loss.item()
        correct, samples = accuracy(outputs, batch_y)
        total_correct += correct
        total_samples += samples

    avg_loss = total_loss / len(loader)
    avg_acc = total_correct / total_samples
    return avg_loss, avg_acc

best_val_acc = 0.0
train_loss_list = []
training_accuracy_list = []
val_loss_list = []
val_acc_list = []
for epoch in range(config.EPOCHS):
    train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
    val_loss, val_acc = validate(model, val_loader, criterion, device) 

    train_loss_list.append(train_loss)
    training_accuracy_list.append(train_acc)
    val_acc_list.append(val_acc)
    val_loss_list.append(val_loss) 

    print(
        f"Epoch [{epoch+1}/{config.EPOCHS}] "
        f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} "
        f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}"
    )

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        torch.save(model.state_dict(), config.MODEL_SAVE_PATH)

print(f"\nBest Validation Accuracy: {best_val_acc: .4f}")
print(f"Model saved to: {config.MODEL_SAVE_PATH}")

