import torch
import segmentation_models_pytorch as smp
from torchvision import transforms
from PIL import Image
from pathlib import Path
import numpy as np
import cv2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load model (DeepLabv3+)
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    in_channels=3,
    classes=1,
    encoder_weights=None
)
checkpoint = torch.load("checkpoints/deeplabv3plus_resnet101_epoch30.pth", map_location=device)
model.load_state_dict(checkpoint["model"])  # ✅ ambil hanya state_dict model
model.to(device)
model.eval()

# Preprocessing
# Preprocessing (samakan dengan train/test)
preprocess = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor()
])


def clean_mask(mask_np, min_area=50):
    mask_np = (mask_np * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cleaned = np.zeros_like(mask_np)
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(cleaned, [cnt], -1, 255, -1)
    return cleaned

def overlay_mask(image_pil, mask_np, color=(255, 0, 0), alpha=0.5):
    image = np.array(image_pil.convert("RGB"))
    mask = (mask_np > 0).astype(np.uint8)
    overlay = image.copy()
    overlay[mask == 1] = color
    blended = cv2.addWeighted(overlay, alpha, image, 1 - alpha, 0)
    return Image.fromarray(blended)

def predict_mask(image_path: str, save_dir: str = "output", threshold: float = 0.5):
    img = Image.open(image_path).convert("RGB")
    input_tensor = preprocess(img).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred)
        pred_mask = (pred > threshold).float().cpu().squeeze().numpy()

    # resize mask ke ukuran asli
    mask_resized = cv2.resize(pred_mask, img.size, interpolation=cv2.INTER_NEAREST)
    cleaned = clean_mask(mask_resized, min_area=100)
    overlay_img = overlay_mask(img, cleaned)

    # simpan hasil
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    mask_path = Path(save_dir) / "mask.png"
    overlay_path = Path(save_dir) / "overlay.png"

    Image.fromarray(cleaned).save(mask_path)
    overlay_img.save(overlay_path)

    return str(mask_path), str(overlay_path)