# import os
# import json
# import io
# from typing import Optional

# from fastapi import FastAPI, UploadFile, File, HTTPException
# from fastapi.middleware.cors import CORSMiddleware
# from PIL import Image
# import torch
# from transformers import ViTForImageClassification, ViTImageProcessor


# # ----------------------------
# # CONFIG
# # ----------------------------
# MODEL_DIR = "models/vit"    # folder that contains model.safetensors + config.json
# LABELS_PATH = "frontend/public/labels.json"
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# app = FastAPI(title="PROJECT_AI Inference API")

# app.add_middleware(
#     CORSMiddleware,
#     allow_origins=["*"],
#     allow_credentials=True,
#     allow_methods=["*"],
#     allow_headers=["*"],
# )


# # ----------------------------
# # Load labels.json
# # ----------------------------
# labels = {}
# if os.path.exists(LABELS_PATH):
#     with open(LABELS_PATH, "r", encoding="utf-8") as f:
#         labels = json.load(f)


# # ----------------------------
# # MODEL LOADER (lazy)
# # ----------------------------
# _model = None
# _processor = None


# def load_model_and_processor():
#     global _model, _processor

#     if _model is not None and _processor is not None:
#         return _model, _processor

#     print("🔄 Loading model...")

#     _processor = ViTImageProcessor.from_pretrained(MODEL_DIR)

#     _model = ViTForImageClassification.from_pretrained(
#         MODEL_DIR,
#         num_labels=101,
#         ignore_mismatched_sizes=True,
#     )

#     _model.to(DEVICE)
#     _model.eval()

#     print("✅ Model loaded successfully!")
#     return _model, _processor


# # ----------------------------
# # ROUTES
# # ----------------------------
# @app.get("/")
# def root():
#     return {"msg": "backend is alive"}


# @app.post("/predict/")
# async def predict(file: UploadFile = File(...)):
#     if not file.content_type or not file.content_type.startswith("image/"):
#         raise HTTPException(status_code=400, detail="File must be an image")

#     contents = await file.read()
#     try:
#         image = Image.open(io.BytesIO(contents)).convert("RGB")
#     except Exception:
#         raise HTTPException(status_code=400, detail="Invalid image")

#     model, processor = load_model_and_processor()

#     inputs = processor(images=image, return_tensors="pt")
#     inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

#     with torch.no_grad():
#         outputs = model(**inputs)
#         logits = outputs.logits
#         probs = torch.softmax(logits, dim=-1)
#         topk = torch.topk(probs, k=1, dim=-1)
#         class_index = int(topk.indices[0][0].item())
#         confidence = float(topk.values[0][0].item())

#     # ✅ FIXED: correct label lookup for your list format JSON
#     raw_name = labels[class_index] if isinstance(labels, list) else labels.get(str(class_index))
#     display_name = raw_name.replace("_", " ") if raw_name else f"class_{class_index}"

#     print(f"✅ Prediction: {display_name} (class {class_index}) conf={confidence:.4f}")

#     return {
#         "prediction": {
#             "class_index": class_index,
#             "rawName": raw_name or display_name,
#             "confidence": confidence,
#         }
#     }




import os
import json
import io
from typing import Optional

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from PIL import Image
import torch
from transformers import ViTForImageClassification, ViTImageProcessor


# ----------------------------
# CONFIG
# ----------------------------
MODEL_DIR = "models/vit"    # folder that contains model.safetensors + config.json
LABELS_PATH = "frontend/public/labels.json"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = FastAPI(title="PROJECT_AI Inference API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ----------------------------
# Load labels.json
# ----------------------------
labels = {}
if os.path.exists(LABELS_PATH):
    with open(LABELS_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)


# ----------------------------
# MODEL LOADER (lazy)
# ----------------------------
_model = None
_processor = None


def load_model_and_processor():
    global _model, _processor

    if _model is not None and _processor is not None:
        return _model, _processor

    print("🔄 Loading model...")

    # Load processor config (already correct)
    _processor = ViTImageProcessor.from_pretrained(MODEL_DIR)

    # Load base architecture
    _model = ViTForImageClassification.from_pretrained(
        MODEL_DIR,
        num_labels=101,
        ignore_mismatched_sizes=True,
    )

    # ✅ Load your trained Food-101 checkpoint
    pth_path = "models/food101_vit_base_patch16_224.pth"

    if os.path.exists(pth_path):
        checkpoint = torch.load(pth_path, map_location=DEVICE)

        # Handle both checkpoint formats
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
            checkpoint = checkpoint["model_state_dict"]

        _model.load_state_dict(checkpoint, strict=False)
        print("✅ Loaded trained .pth weights!")
    else:
        print("⚠️ .pth file not found, using default base model instead.")

    _model.to(DEVICE)
    _model.eval()

    return _model, _processor


# ----------------------------
# ROUTES
# ----------------------------
@app.get("/")
def root():
    return {"msg": "backend is alive"}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="File must be an image")

    contents = await file.read()
    try:
        image = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid image")

    model, processor = load_model_and_processor()

    inputs = processor(images=image, return_tensors="pt")
    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probs = torch.softmax(logits, dim=-1)
        topk = torch.topk(probs, k=1, dim=-1)
        class_index = int(topk.indices[0][0].item())
        confidence = float(topk.values[0][0].item())

    # ✅ FIXED: correct label lookup for your list format JSON
    raw_name = labels[class_index] if isinstance(labels, list) else labels.get(str(class_index))
    display_name = raw_name.replace("_", " ") if raw_name else f"class_{class_index}"

    print(f"✅ Prediction: {display_name} (class {class_index}) conf={confidence:.4f}")

    return {
        "prediction": {
            "class_index": class_index,
            "rawName": raw_name or display_name,
            "confidence": confidence,
        }
    }
