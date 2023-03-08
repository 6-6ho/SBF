import torch.nn as nn
import torch
from data_loader import CustomImageDataset
from model import ResNet
from torch.utils.data import DataLoader

epochs = 10
batch_size = 32
learning_rate = 0.0001

train_loader = DataLoader(CustomImageDataset("train"), batch_size=batch_size, shuffle=True)
valid_loader = DataLoader(CustomImageDataset("valid"), batch_size=batch_size, shuffle=True)
test_loader = DataLoader(CustomImageDataset("test"), batch_size=batch_size, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
model = ResNet(num_classes=3).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

for epoch in range(epochs):
    model.train()
    train_loss = 0.0
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item() * images.size(0)
    train_loss /= len(train_loader.dataset)
    
    model.eval()
    valid_loss = 0.0
    valid_correct = 0
    with torch.no_grad():
        for images, labels in valid_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            valid_loss += loss.item() * images.size(0)
            _, preds = torch.max(outputs, 1)
            valid_correct += torch.sum(preds == labels.data)
    valid_loss /= len(valid_loader.dataset)
    valid_acc = valid_correct.double() / len(valid_loader.dataset)
    
    print(f'Epoch {epoch+1}/{epochs} -- '
          f'Training Loss: {train_loss:.4f} -- '
          f'Validation Loss: {valid_loss:.4f} -- '
          f'Validation Accuracy: {valid_acc:.4f}')

model.eval()
test_correct = 0
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, preds = torch.max(outputs, 1)
        test_correct += torch.sum(preds == labels.data)
test_acc = test_correct.double() / len(test_loader.dataset)
print(f'Test Accuracy: {test_acc:.4f}')

torch.save(model.state_dict(), 'resnet_model.pth')
