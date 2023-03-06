import torch.nn as nn
import torch
from data_loader import CustomImageDataset
from model import ResNet
from torch.utils.data import DataLoader

classes = ['beuk', 'chung', 'hye']

my_epoch = 15
my_batch = 32
my_learning_rate = 0.0001

print("epoch :", my_epoch)
print("batch_size :", my_batch)
print("learning_rate :", my_learning_rate)

train_data_set = CustomImageDataset("train")
train_loader = DataLoader(train_data_set, batch_size=my_batch, shuffle=True)

valid_data_set = CustomImageDataset("valid")
valid_loader = DataLoader(valid_data_set, batch_size=my_batch, shuffle=True)

test_data_set = CustomImageDataset("test")
test_loader = DataLoader(test_data_set, batch_size=my_batch, shuffle=True)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

model = ResNet(num_classes=len(classes)).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=my_learning_rate)

# Train
for epoch in range(my_epoch):
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
    
    # Evaluate validation set
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
    
    print(f'Epoch {epoch+1}/{my_epoch} -- '
          f'Training Loss: {train_loss:.4f} -- '
          f'Validation Loss: {valid_loss:.4f} -- '
          f'Validation Accuracy: {valid_acc:.4f}')

# Evaluate test set
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
