import os
import torch
import torchvision.transforms as transforms
from torchvision import models
from PIL import Image
import numpy as np
import faiss

# --- STEP 1: Load Pre-trained Model ---
# Sử dụng mô hình ResNet50 để trích xuất đặc trưng
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = models.resnet50(pretrained=True)
feature_extractor = torch.nn.Sequential(*list(model.children())[:-1])  # Bỏ lớp classification cuối
feature_extractor = feature_extractor.to(DEVICE)
feature_extractor.eval()  # Đặt mô hình vào chế độ inference

# --- STEP 2: Preprocessing (Resize và Normalize ảnh) ---
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

def load_image(image_path):
    image = Image.open(image_path).convert("RGB")  # Load ảnh và chuyển sang RGB
    image = transform(image).unsqueeze(0)  # Thêm batch dimension
    return image

# --- STEP 3: Feature Extraction ---
def extract_features(image_path):
    """Trích xuất vector đặc trưng từ ảnh."""
    image = load_image(image_path).to(DEVICE)
    with torch.no_grad():
        features = feature_extractor(image)
        features = features.flatten().cpu().numpy()  # Chuyển thành vector 1D
    return features

# --- STEP 4: Dataset và FAISS ---
# Tạo FAISS Index để lưu trữ vector
def create_faiss_index(dataset_path):
    """Duyệt qua dataset, trích xuất đặc trưng và lưu vào FAISS."""
    d = 2048  # Kích thước vector đặc trưng (phụ thuộc vào mô hình ResNet50)
    index = faiss.IndexFlatL2(d)  # Sử dụng Euclidean distance
    product_vectors = []  # Lưu vector đặc trưng
    product_ids = []      # Lưu product ID
    image_paths = []      # Đường dẫn ảnh

    for image_name in os.listdir(dataset_path):
        image_path = os.path.join(dataset_path, image_name)
        product_id = image_name.split(".")[0]  # Lấy tên file làm ID sản phẩm
        vector = extract_features(image_path)
        product_vectors.append(vector)
        product_ids.append(product_id)
        image_paths.append(image_path)

    # Convert dữ liệu sang NumPy array
    product_vectors = np.array(product_vectors, dtype="float32")
    index.add(product_vectors)  # Thêm vector vào FAISS index

    return index, product_ids, image_paths

# --- STEP 5: Tìm kiếm sản phẩm tương tự ---
def search_similar_products(image_path, index, product_ids, image_paths, top_k=5):
    """Tìm sản phẩm tương tự dựa trên ảnh upload."""
    query_vector = extract_features(image_path).astype("float32")
    distances, indices = index.search(query_vector.reshape(1, -1), top_k)  # Tìm top_k vector gần nhất
    similar_products = [{"id": product_ids[i], "path": image_paths[i], "distance": distances[0][idx]}
                        for idx, i in enumerate(indices[0])]
    return similar_products

# --- STEP 6: Hiển thị kết quả ---
def display_results(results):
    """Hiển thị sản phẩm tương tự."""
    print("\n--- KẾT QUẢ TÌM KIẾM ---")
    for result in results:
        print(f"Product ID: {result['id']}, Distance: {result['distance']:.4f}")
        print(f"Image Path: {result['path']}")

# --- MAIN FUNCTION ---
if __name__ == "__main__":
    # Đường dẫn đến dataset
    dataset_path = "D:\\doan\\image-search\\course\\DataTrain\\train\\images"  # Thư mục chứa ảnh sản phẩm
    query_image_path = ""  # Ảnh mà user upload

    # Tạo FAISS index từ dataset
    print(">>> Trích xuất vector đặc trưng và tạo FAISS index...")
    index, product_ids, image_paths = create_faiss_index(dataset_path)
    print(">>> Tạo FAISS index hoàn tất!")

    # Tìm kiếm sản phẩm tương tự
    print(f">>> Tìm kiếm sản phẩm tương tự cho ảnh: {query_image_path}")
    results = search_similar_products(query_image_path, index, product_ids, image_paths)

    # Hiển thị kết quả
    display_results(results)
