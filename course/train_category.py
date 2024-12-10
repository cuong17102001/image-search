import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms, models, datasets

# Configurations
BATCH_SIZE = 32
EPOCHS = 10
LEARNING_RATE = 0.001
NUM_CLASSES = 10  # Số lượng nhãn trong dataset
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Step 1: Preprocessing
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),  # Augmentation
    transforms.RandomRotation(15),     # Augmentation
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize
])

train_dataset ="D:\\doan\\image-search\\course\DataTrain\\train\\images"
val_dataset = "D:\doan\image-search\course\DataTrain\\validation\images"
# Step 2: Load Dataset
train_dataset = datasets.ImageFolder(root=train_dataset, transform=transform)
val_dataset = datasets.ImageFolder(root=val_dataset, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Step 3: Define Model
model = models.resnet50(pretrained=True)
model.fc = nn.Linear(model.fc.in_features, NUM_CLASSES)  # Replace final layer
model = model.to(DEVICE)

# Step 4: Loss Function and Optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# Step 5: Training Loop
def train_model():
    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0

        for images, labels in train_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Backward pass and optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

        print(f"Epoch [{epoch + 1}/{EPOCHS}], Loss: {running_loss / len(train_loader):.4f}")

        # Validation
        validate_model()

def validate_model():
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in val_loader:
            images, labels = images.to(DEVICE), labels.to(DEVICE)

            outputs = model(images)
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f"Validation Accuracy: {100 * correct / total:.2f}%")

# Step 6: Save Model
def save_model(path="model.pth"):
    torch.save(model.state_dict(), path)
    print("Model saved!")

# Run Training
train_model()
save_model()
