import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from torchvision import models
from torch.utils.data import DataLoader
from torchvision.models import ResNet50_Weights

# 1. Dataset Directory
DATA_DIR = "D:\\doan\\image-search\\course\\DataTrain"  # Thay thế bằng đường dẫn dataset của bạn
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# 2. Hyperparameters
BATCH_SIZE = 16
EPOCHS = 5
LEARNING_RATE = 0.001
NUM_CLASSES = 45  # Áo, Quần, Giày

# 3. Image Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 4. Dataset & DataLoader
train_dataset = datasets.ImageFolder(root=os.path.join("D:\\doan\\e-commerce-mern\\server\\public\\uploads\\products"), transform=transform)
val_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "val"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# 5. Model Definition (Pretrained ResNet50)
model = models.resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
for param in model.parameters():
    param.requires_grad = False  # Freeze pre-trained weights

# Replace the final classification layer
num_ftrs = model.fc.in_features
model.fc = nn.Linear(num_ftrs, NUM_CLASSES)
model = model.to(DEVICE)

# 6. Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.fc.parameters(), lr=LEARNING_RATE)

# 7. Train Model
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs):
    epochNumber = 0
    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        imageNumber = 0
        for images, labels in train_loader:
            print("epoch:" + str(epochNumber) + ", image number:" + str(imageNumber))
            imageNumber += 1
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
        epochNumber += 1
        print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")
    return model

print("Training model...")
model = train_model(model, train_loader, val_loader, criterion, optimizer, EPOCHS)

# Save model
torch.save(model.state_dict(), "product_classifier.pth")
print("Model saved as 'product_classifier.pth'.")
