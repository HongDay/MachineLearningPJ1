import torch
import torch.nn as nn

# test_lodaer : 
def eval_accuracy(test_loader, net, device):
    net.eval()
    correct = 0
    total = 0
    valid_loss = 0
    criterion = nn.CrossEntropyLoss()
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            loss = criterion(outputs, labels)
            valid_loss += loss.item()
    valid_accuracy = (100 * correct / total)
    print(f'Accuracy of the network on the {len(test_loader)*32} test images: {valid_accuracy:.2f}% / loss : {valid_loss:.2f}')


# confusion matrix :

import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

def plot_confusion_matrix(test_loader, net, device, classes):
    net.eval()
    valid_labels = []
    valid_preds = []

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = net(images)
            _, predicted = torch.max(outputs, 1)
            valid_labels.extend(labels.cpu().numpy())
            valid_preds.extend(predicted.cpu().numpy())

    cm = confusion_matrix(valid_labels, valid_preds)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    disp.plot(cmap=plt.cm.Blues)
    plt.title("Confusion Matrix")
    plt.show()


# visualize :

import matplotlib.pyplot as plt

def plot_training_curves(train_loss, valid_loss, train_acc, valid_accuracy, epochs):
    epoch_list = range(1, epochs + 1)

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
    plt.plot(epoch_list, valid_accuracy, label='validation accuracy', color='tab:orange')
    plt.title('Accuracy Trend')
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.legend()
    plt.grid(True)
    plt.show()


https://velog.io/@smile_b/CNN-24-Confusion-Matrix-and-Classification-Report




