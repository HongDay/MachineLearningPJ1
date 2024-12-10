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

