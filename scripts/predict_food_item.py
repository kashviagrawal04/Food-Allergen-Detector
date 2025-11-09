import argparse
import os
import torch
from torchvision import transforms
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor
import json  # 👈 added

MODEL_PATH = "/content/alldete/models/food101_vit_base_patch16_224.pth"
DATA_TRAIN_DIR = "/content/alldete/data/food-101/train"   # change if path is different

# Build class list from folder names
class_names = sorted(os.listdir(DATA_TRAIN_DIR))

# Load image processor and model
processor = ViTImageProcessor.from_pretrained("google/vit-base-patch16-224")

model = ViTForImageClassification.from_pretrained(
    "google/vit-base-patch16-224",
    num_labels=len(class_names),
    ignore_mismatched_sizes=True
)

# Load trained weights
state_dict = torch.load(MODEL_PATH, map_location=torch.device("cpu"))
model.load_state_dict(state_dict, strict=False)
model.eval()

# Transform image
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=processor.image_mean, std=processor.image_std)
])

# Load allergen data 👇
with open("/content/alldete/allergens.json", "r") as f:
    allergen_data = json.load(f)

def predict(image_path):
    print(f"\n🔍 Loading image: {image_path}")
    image = Image.open(image_path).convert("RGB")
    inputs = transform(image).unsqueeze(0)

    with torch.no_grad():
        logits = model(inputs).logits
        predicted_index = logits.argmax(-1).item()

    food_name = class_names[predicted_index]
    print(f"✅ Prediction: {food_name}  (class ID {predicted_index})")

    # Allergen checking 👇
    if food_name in allergen_data:
        user_allergen = input("\nEnter an allergen to check (e.g. dairy, nuts, gluten): ").lower().strip()
        allergens = allergen_data[food_name].get("allergens", [])
        if user_allergen in [a.lower() for a in allergens]:

            print(f"⚠️ Warning: {food_name} contains {user_allergen}!")
        else:
            print(f"✅ Safe: {food_name} does not contain {user_allergen}.")
    else:
        print(f"ℹ️ No allergen data found for '{food_name}'.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    args = parser.parse_args()
    predict(args.image)
