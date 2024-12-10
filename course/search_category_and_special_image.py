import os
import torch
import numpy as np
from torchvision import models, transforms
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity

# 1. Dataset Directory and Model Path
DATA_DIR = "D:\\final project\\e-commerce-mern\\server\\public\\uploads\\products"  # Dataset folder containing subfolders for each category
MODEL_PATH = "product_classifier.pth"  # Path to the trained model
DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 2. Image Transforms
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

# 3. Load Trained Model
def load_feature_extractor(model_path):
    model = models.resnet50(weights='IMAGENET1K_V1')
    model.fc = torch.nn.Identity()  # Replace the last fully connected layer
    state_dict = torch.load(model_path, map_location=DEVICE)
    
    model.load_state_dict(state_dict, strict=False)  # Load the state_dict
    model = model.to(DEVICE)  # Move model to the correct device (GPU/CPU)
    model.eval()

    return model

# 4. Feature Extraction Function (extract features from an image)
def extract_features(model, image):
    """
    Extract feature vector from an image using the trained model.
    """
    model.eval()
    with torch.no_grad():
        features = model(image)
    return features.cpu().numpy()

# 5. Function to Calculate Cosine Similarity
def calculate_cosine_similarity(query_features, product_features):
    return cosine_similarity(query_features, product_features)

# 6. Find Similar Products using Cosine Similarity
def find_similar_products(query_image_path, data_dir, feature_extractor):
    query_image = Image.open(query_image_path).convert("RGB")
    query_image = transform(query_image).unsqueeze(0).to(DEVICE)
    query_features = extract_features(feature_extractor, query_image)

    # Collect image paths for the dataset
    print("Collecting image paths from dataset...")
    product_paths = []
    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            product_paths.extend([os.path.join(class_path, img_name) for img_name in os.listdir(class_path)])

    # Load precomputed product features from the .npy file
    print("Loading product features from .npy file...")
    product_features = np.load('product_features.npy')
    print(f"Loaded product features with shape: {product_features.shape}")

    # Calculate cosine similarity between the query image and all product images
    print("Calculating cosine similarities...")
    similarities = calculate_cosine_similarity(query_features, product_features)

    # Sort products by similarity (descending order)
    sorted_indices = np.argsort(similarities[0])[::-1]
    top_k_indices = sorted_indices[:20]  # Get top 20 most similar products

    # Get image paths of the top similar products
    similar_images = [product_paths[i] for i in top_k_indices]
    
    print("Search complete.")
    return similar_images

# 7. Query Example
if __name__ == '__main__':
    query_image_path = "C:\\Users\\ndhoa\\Downloads\\7e63902d33109ca1701038cc2c8e522a.jpg"  # Replace with your query image path
    feature_extractor = load_feature_extractor(MODEL_PATH)
    print("Device of feature extractor: " + str(next(feature_extractor.parameters()).device))

    similar_products = find_similar_products(query_image_path, DATA_DIR, feature_extractor)
    print(similar_products)

    # Show query and similar images
    query_image = Image.open(query_image_path)
    plt.figure(figsize=(15, 10))  # Adjust the figure size to accommodate more images
    plt.subplot(1, 21, 1)
    plt.imshow(query_image)
    plt.title("Query Image")
    plt.axis("off")

    for i, img_path in enumerate(similar_products):
        img = Image.open(img_path)
        plt.subplot(1, 21, i + 2)
        plt.imshow(img)
        plt.title(f"Similar {i+1}")
        plt.axis("off")
    plt.show()
