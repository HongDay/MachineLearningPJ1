import torch
import torch.nn as nn

# test_lodaer : 
def eval_accuracy(valid_loader, model, device):
    model.eval()
    valid_correct = 0
    valid_total = 0
    valid_loss = 0
    criterion = nn.CrossEntropyLoss()

    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            valid_total += target.size(0)
            valid_correct += (predicted == target).sum().item()
            loss = criterion(output, target)
            valid_loss += loss.item()

    valid_accuracy = (100 * valid_correct / valid_total)
    print(f'Validation Accuracy: {valid_accuracy:.2f}% / Validation Loss: {valid_loss:.2f}')



# confusion matrix :

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(valid_loader, model, device, classes):
    model.eval()
    valid_labels = []
    valid_preds = []

    with torch.no_grad():
        for data, target in valid_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            _, predicted = torch.max(output, 1)
            valid_labels.extend(target.cpu().numpy())
            valid_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(valid_labels, valid_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()



# visualize :

import matplotlib.pyplot as plt

def plot_training_curves(train_loss, valid_loss, train_acc, valid_acc, n_epochs):
    epoch_list = range(1, n_epochs + 1)

    plt.figure(figsize=(8, 6))
    plt.plot(epoch_list, train_loss, label='train loss', color='tab:blue')
    plt.plot(epoch_list, valid_loss, label='validation loss', color='tab:orange')
    plt.title('Loss Trend')
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.legend()
    plt.grid(True)
    plt.show()

    plt.figure(figsize=(8, 6))
    plt.plot(epoch_list, train_acc, label='train accuracy', color='tab:blue')
    plt.plot(epoch_list, valid_acc, label='validation accuracy', color='tab:orange')
    plt.title('Accuracy Trend')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


https://velog.io/@smile_b/CNN-24-Confusion-Matrix-and-Classification-Report




