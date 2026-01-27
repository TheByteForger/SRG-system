import matplotlib.pyplot as plt
import train

train_loss = train.train_loss_list

val_loss =  train.val_loss_list

train_acc = train.training_accuracy_list

val_acc =   train.val_acc_list

epochs = range(1, len(train_loss)+1)

plt.figure(figsize=(12,5))
plt.subplot(1,2,1)
plt.plot(epochs, train_loss, 'b-o', label='Train Loss')
plt.plot(epochs, val_loss, 'r-o', label='Val Loss')
plt.title('Loss vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1,2,2)
plt.plot(epochs, train_acc, 'b-o', label='Train Acc')
plt.plot(epochs, val_acc, 'r-o', label='Val Acc')
plt.title('Accuracy vs Epochs')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()
