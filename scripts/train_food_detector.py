import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from torchvision.datasets import Food101
from transformers import ViTForImageClassification, ViTImageProcessor
from tqdm import tqdm

# === Paths ===
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "../data/food-101")
MODEL_SAVE_PATH = os.path.join(BASE_DIR, "../models/food101_vit_base_patch16_224.pth")

# === Device setup ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"✅ Using device: {device}")

# === Check / Download dataset ===
if not os.path.exists(os.path.join(DATA_DIR, "train")):
    print("📦 Food-101 dataset not found — downloading automatically...")
    os.makedirs(os.path.dirname(DATA_DIR), exist_ok=True)
    _ = Food101(root=os.path.dirname(DATA_DIR), download=True)
else:
    print("✅ Found existing Food-101 dataset.")

# === Processor & transforms ===
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# === Datasets & loaders ===
train_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "train"), transform=transform)
test_dataset = datasets.ImageFolder(root=os.path.join(DATA_DIR, "test"), transform=transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

# === Model ===
model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(train_dataset.classes),
    ignore_mismatched_sizes=True
).to(device)

# === Training setup ===
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

# === Training loop ===
epochs = 2
for epoch in range(epochs):
    model.train()
    total_loss = 0
    for images, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}"):
        images, labels = images.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(images).logits
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"📉 Epoch [{epoch+1}/{epochs}] - Loss: {total_loss/len(train_loader):.4f}")

# === Save model ===
os.makedirs(os.path.dirname(MODEL_SAVE_PATH), exist_ok=True)
torch.save(model.state_dict(), MODEL_SAVE_PATH)
print(f"✅ Model saved to {MODEL_SAVE_PATH}")
