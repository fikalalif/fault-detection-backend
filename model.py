import torch
import segmentation_models_pytorch as smp
from PIL import Image
from pathlib import Path
import numpy as np
import cv2
import os

# --- Import BARU untuk Augmentasi ---
import albumentations as A
from albumentations.pytorch import ToTensorV2

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Model.py using device: {device}")

# --- FIX 1: Tentukan Path & Encoder yang BENAR ---
MODEL_ENCODER = "resnet101"
MODEL_PATH = Path("models/deeplabv3plus_resnet101_colab.pth")
IMG_HEIGHT = 512
IMG_WIDTH = 512

# Load model (DeepLabv3+)
model = smp.DeepLabV3Plus(
    encoder_name=MODEL_ENCODER,
    in_channels=3,
    classes=1,
    encoder_weights=None
)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    print(f"Model {MODEL_PATH} loaded.")
except Exception as e:
    print(f"!!! ERROR loading model.py: {e} !!!")
    print("Pastikan file model ada di path yang benar.")
    
model.to(device)
model.eval()

# --- FIX 2: Ganti preprocessing ke Albumentations (HARUS SAMA DENGAN VALIDASI) ---
preprocess = A.Compose(
    [
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(),
    ]
)

# --- FUNGSI clean_mask (DI-UPDATE) ---
def clean_mask(mask_np, min_area=50):
    """
    Membersihkan noise kecil dari mask.
    Mengembalikan mask yang sudah bersih DAN jumlah retakan yang valid.
    """
    mask_np_u8 = (mask_np * 255).astype(np.uint8)
    contours, _ = cv2.findContours(mask_np_u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cleaned = np.zeros_like(mask_np_u8)
    valid_contour_count = 0 # <-- BARU
    
    for cnt in contours:
        if cv2.contourArea(cnt) > min_area:
            cv2.drawContours(cleaned, [cnt], -1, 255, -1)
            valid_contour_count += 1 # <-- BARU
            
    cleaned_binary = (cleaned > 0).astype(np.uint8)
    return cleaned_binary, valid_contour_count # <-- BARU: Mengembalikan 2 nilai


def overlay_mask(image_pil, mask_np, color=(255, 0, 0), alpha=0.5):
    """Membuat overlay mask di atas gambar asli."""
    image = np.array(image_pil.convert("RGB"))
    mask = (mask_np > 0).astype(np.uint8)
    
    mask_rgb = np.zeros_like(image)
    mask_rgb[mask == 1] = color
    
    blended = cv2.addWeighted(mask_rgb, alpha, image, 1 - alpha, 0)
    blended[mask == 0] = image[mask == 0]
    
    return Image.fromarray(blended)

# --- FUNGSI predict_mask (DI-UPDATE) ---
def predict_mask(image_path: str, save_dir: str = "output", threshold: float = 0.5):
    try:
        img_pil = Image.open(image_path).convert("RGB")
        img_np = np.array(img_pil) 
    except Exception as e:
        print(f"Error membuka gambar: {e}")
        return None, None, None # <-- BARU: Kembalikan 3 nilai

    # Preprocessing
    augmented = preprocess(image=img_np)
    input_tensor = augmented['image'].unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(input_tensor)
        pred = torch.sigmoid(pred)
        
        pred_mask_resized = torch.nn.functional.interpolate(
            pred,
            size=(img_pil.height, img_pil.width),
            mode='bilinear',
            align_corners=False
        )
        pred_mask_np = (pred_mask_resized > threshold).float().cpu().squeeze().numpy()

    # (Opsional) Bersihkan mask
    # --- BARU: Tangkap jumlah retakan (contour_count) ---
    cleaned_mask_np, contour_count = clean_mask(pred_mask_np, min_area=10)

    # ==================================
    # --- LOGIKA TEKS DINAMIS (BARU) ---
    # ==================================
    total_pixels = img_pil.height * img_pil.width
    fault_pixels = cleaned_mask_np.sum()
    fault_percentage = (fault_pixels / total_pixels) * 100

    summary_text = ""
    if contour_count == 0:
        summary_text = "Status: AMAN. Tidak terdeteksi adanya retakan (sesar) yang signifikan. Kondisi terlihat stabil."
    elif fault_percentage < 0.75 and contour_count < 10:
        summary_text = f"Status: WASPADA. Terdeteksi {contour_count} retakan minor (mencakup {fault_percentage:.2f}% dari total area). Disarankan pemantauan berkala."
    else:
        summary_text = f"Status: PERINGATAN. Terdeteksi {contour_count} retakan signifikan (mencakup {fault_percentage:.2f}% dari total area). Indikasi kuat adanya aktivitas sesar. Disarankan inspeksi lebih lanjut oleh ahli geologi."
    # ==================================
    # --- AKHIR LOGIKA TEKS ---
    # ==================================

    # Simpan file mask
    mask_img = Image.fromarray((cleaned_mask_np * 255).astype(np.uint8))
    mask_path = os.path.join(save_dir, "mask.png")
    mask_img.save(mask_path)

    # Simpan file overlay
    overlay_img = overlay_mask(img_pil, cleaned_mask_np)
    overlay_path = os.path.join(save_dir, "overlay.png")
    overlay_img.save(overlay_path)

    return mask_path, overlay_path, summary_text # <-- BARU: Kembalikan 3 nilai