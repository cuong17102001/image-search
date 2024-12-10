import torch
import torch.nn as nn
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import os
import numpy as np

# Định nghĩa hàm trích xuất đặc trưng từ ảnh
def extract_features(image_path, model, transform):
    image = Image.open(image_path)
    image = transform(image).unsqueeze(0)  # Thêm batch dimension
    
    # Di chuyển ảnh và mô hình sang GPU nếu có
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    image = image.to(device)
    
    with torch.no_grad():  # Tắt tính toán gradient để tiết kiệm bộ nhớ
        features = model(image)
    
    return features.cpu().numpy()

# Tải mô hình ResNet50 đã huấn luyện trước
model = models.resnet50(pretrained=True)
model = nn.Sequential(*list(model.children())[:-1])  # Loại bỏ layer phân loại cuối

# Định nghĩa các phép biến đổi ảnh
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Trích xuất đặc trưng từ ảnh
print("lấy tất cả đường dẫn ảnh từ dataset")
DATA_DIR = "D:\\final project\\e-commerce-mern\\server\\public\\uploads\\products"
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
product_paths = []
for class_dir in os.listdir(DATA_DIR):
    class_path = os.path.join(DATA_DIR, class_dir)
    if os.path.isdir(class_path):
        product_paths.extend([os.path.join(class_path, img_name) for img_name in os.listdir(class_path)])

# Trích xuất đặc trưng cho tất cả ảnh trong dataset
print("Trích xuất đặc trưng cho tất cả ảnh trong dataset")
product_features = []
product_image_paths = []
for image_path in product_paths:
    print(image_path)
    features = extract_features(image_path, model, transform)
    
    # Flatten the features to a 1D vector (shape: 2048,)
    features = features.flatten()  # Chuyển đổi từ (1, 2048, 1, 1) thành 2048
    product_features.append(features)
    product_image_paths.append(image_path)

# Chuyển danh sách thành mảng NumPy 2D
product_features = np.array(product_features)

# Lưu lại các đặc trưng vào file numpy
np.save('product_features.npy', product_features)