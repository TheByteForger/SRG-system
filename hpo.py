import optuna
import torch
import torch.nn as nn
import torch.optim as optim
import src.config as config
from src.model import HomeAudioCNN
from dataset_loader import get_dataloaders
import os

def objective(trial):
 
    lr = trial.suggest_float("lr", 1e-4, 5e-3, log=True)
    base_filters = trial.suggest_categorical("base_filters", [4, 8, 12, 16])
    dropout_rate = trial.suggest_float("dropout_rate", 0.2, 0.6)
    batch_size = trial.suggest_categorical("batch_size", [16, 32])
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)

    device = config.DEVICE
    train_loader, val_loader, _ = get_dataloaders(batch_size=batch_size)
    
    model = HomeAudioCNN(
        num_classes=config.NUM_CLASSES, 
        base_filters=base_filters, 
        dropout_rate=dropout_rate
    ).to(device)

    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.CrossEntropyLoss()

    best_trial_acc = 0
    for epoch in range(15):
        model.train()
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            output = model(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()

        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                output = model(batch_X)
                _, predicted = torch.max(output.data, 1)
                total += batch_y.size(0)
                correct += (predicted == batch_y).sum().item()
        
        accuracy = correct / total

        trial.report(accuracy, epoch)
        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()
            
        best_trial_acc = max(best_trial_acc, accuracy)

    return best_trial_acc

if __name__ == "__main__":
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=30)

    print("\n--- Optimization Finished ---")
    print(f"Best value (Accuracy): {study.best_value:.4f}")
    print("Best parameters:")
    for key, value in study.best_params.items():
        print(f"    {key}: {value}")
