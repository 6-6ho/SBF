import torch.nn as nn
import torch
from data_loader import CustomImageDataset
from model import ResNet
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

def prediction():
    classes = ['beuk', 'chung', 'hye']

    my_epoch = 15
    my_batch = 32
    my_learning_rate = 0.0001

    print("epoch :", my_epoch)
    print("batch_size :", my_batch)
    print("learning_rate :", my_learning_rate)

    train_data_set = CustomImageDataset("train")
    train_loader = DataLoader(train_data_set, batch_size=my_batch, shuffle=True)

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

    # Get the prediciton
    image_path = 'example_img.jpg'
    image = Image.open(image_path)

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    image_tensor = transform(image).unsqueeze(0)

    # Get the model's prediction
    with torch.no_grad():
        model.eval()
        output = model(image_tensor.to(device))
        _, predicted = torch.max(output, 1)
        predicted_label = classes[predicted]

    print(f'The predicted label for this image is: {predicted_label}')
    return predicted_label
