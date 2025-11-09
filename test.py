import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from PIL import Image

# --- Import BARU untuk Augmentasi ---
import albumentations as A
from albumentations.pytorch import ToTensorV2
from dataset_class import FaultDataset # Import class baru dari dataset_class.py
# --- Hapus: import torchvision.transforms as T ---


# ===============================
# 1. KONFIGURASI (Sesuaikan)
# ===============================
DATASET_DIR = Path("/home/fikal/fikal/fault_detection/fault-detection-backend/dataset_fault")
# --- FIX 1: Ganti nama model ke resnet34 ---
MODEL_PATH = Path("models/deeplabv3plus_resnet34_best.pth") 
SAMPLES_DIR = Path("samples_test") # Folder untuk menyimpan hasil gambar tes
SAMPLES_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 4 # Sesuaikan dengan BATCH_SIZE training-mu
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# --- Ukuran Gambar (HARUS SAMA DENGAN SAAT TRAINING) ---
IMG_HEIGHT = 512
IMG_WIDTH = 512

# ===============================
# 2. DEFINISIKAN TRANSFORM (Albumentations)
# ===============================
# Transformasi untuk data TEST (SAMA DENGAN VALIDASI DI train.py)
test_transform = A.Compose(
    [
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(), # Konversi ke Tensor
    ]
)

# ===============================
# 3. LOAD DATASET (Dengan Fungsi fix_path yang BENAR)
# ===============================
print("Loading test dataset...")
patched_images_dir = DATASET_DIR / "patched_images"

try:
    df = pd.read_csv("result_parsing/test.csv")
    print(f"Loaded {len(df)} records from test.csv")
except FileNotFoundError:
    print("Warning: test.csv not found. Falling back to validation.csv")
    df = pd.read_csv("result_parsing/validation.csv")
    print(f"Loaded {len(df)} records from validation.csv")

# --- Fungsi fix_path (FIXED) ---
def fix_path(path_str):
    if isinstance(path_str, str):
        # FIX: Ganti backslash (Windows) ke forward slash (Linux) lalu ambil nama filenya
        filename = path_str.replace("\\", "/").split("/")[-1]
        
        full_img_path = patched_images_dir / filename
        if not full_img_path.exists():
            return None
        return str(full_img_path)
    return str(path_str)

df["image"] = df["Original Image Patch"].apply(fix_path)
df["mask"]  = df["Binary Mask Image Patch"].apply(fix_path)

# --- BARU: Hapus baris yang filenya tidak ditemukan ---
df = df.dropna(subset=["image", "mask"])
print(f"Total Test data loaded: {len(df)} samples")

if len(df) == 0:
    print("!!! ERROR: Dataset kosong. Cek path dan CSV.")
    exit()

# ===============================
# 4. DATASET & DATALOADER (Versi Baru)
# ===============================
dataset = FaultDataset(df, transform=test_transform) # Gunakan transform baru
loader = DataLoader(
    dataset, 
    batch_size=BATCH_SIZE,
    num_workers=4,
    shuffle=False # Jangan di-shuffle saat testing
)

# ===============================
# 5. LOAD MODEL
# ===============================
print(f"Loading model from {MODEL_PATH}")
model = smp.DeepLabV3Plus(
    # --- FIX 2: Ganti encoder ke resnet34 ---
    encoder_name="resnet34",
    in_channels=3,
    classes=1,
    encoder_weights=None # Set ke None karena kita load bobot sendiri
)

try:
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
except Exception as e:
    print(f"Error: Tidak bisa load model dari {MODEL_PATH}.")
    print(f"Detail: {e}")
    print("Pastikan file model ada dan training sudah dijalankan.")
    exit()
        
model.to(device)
model.eval() # Set model ke mode evaluasi (penting!)
print("Model loaded successfully.")

# ===============================
# 6. DEFINISI LOSS & METRICS (Ambil dari train.py)
# ===============================
try:
    from train import combined_loss, calc_metrics
    print("Metrics imported from train.py")
except ImportError:
    print("Warning: Gagal import dari train.py. Mendefinisikan fungsi loss/metrics dummy.")
    
    # Definisi cadangan jika import gagal
    dice_loss = smp.losses.DiceLoss(mode="binary")
    bce_loss = nn.BCEWithLogitsLoss()
    def combined_loss(preds, masks):
        return 0.5 * dice_loss(preds, masks) + 0.5 * bce_loss(preds, masks)

    def calc_metrics(preds, masks, threshold=0.5):
        preds_bin = (torch.sigmoid(preds) > threshold).float()
        masks_bin = masks.float()
        tp = (preds_bin * masks_bin).sum()
        fp = (preds_bin * (1 - masks_bin)).sum()
        fn = ((1 - preds_bin) * masks_bin).sum()
        iou = (tp + 1e-6) / (tp + fp + fn + 1e-6)
        dice = (2 * tp + 1e-6) / (2 * tp + fp + fn + 1e-6)
        return iou.item(), dice.item()

# ===============================
# 7. EVALUATION LOOP
# ===============================
total_loss = 0
total_iou = 0
total_dice = 0
n_batches = len(loader)

print("Starting evaluation...")
# Nonaktifkan perhitungan gradient
with torch.no_grad():
    for i, (imgs, masks) in enumerate(tqdm(loader, desc="Testing")):
        imgs, masks = imgs.to(device), masks.to(device)
        
        preds = model(imgs)

        # Hitung loss
        loss = combined_loss(preds, masks)
        total_loss += loss.item()

        # Hitung metrik
        iou, dice_s = calc_metrics(preds, masks)
        total_iou += iou
        total_dice += dice_s

        # ===============================
        # 8. SIMPAN GAMBAR SAMPEL
        # ===============================
        if i < 5:  
            imgs_np = imgs.cpu().numpy() 
            masks_np = masks.cpu().squeeze(1).numpy()
            preds_np = torch.sigmoid(preds).cpu().squeeze(1).numpy()
            preds_bin_np = (preds_np > 0.5).astype(np.uint8)

            for j in range(len(imgs_np)):
                img = imgs_np[j].transpose(1, 2, 0)
                img = np.clip(img, 0, 1)
                
                mask = masks_np[j]
                pred_bin = preds_bin_np[j]

                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                
                axes[0].imshow(img)
                axes[0].set_title("Image")
                
                axes[1].imshow(mask, cmap="gray")
                axes[1].set_title("Ground Truth (Mask Asli)")
                
                axes[2].imshow(pred_bin, cmap="gray")
                axes[2].set_title("Prediction (Mask Model)")
                
                for ax in axes: ax.axis("off")

                save_path = SAMPLES_DIR / f"sample_batch{i}_img{j}.png"
                plt.savefig(save_path)
                plt.close(fig)

# ===============================
# 9. PRINT HASIL AKHIR
# ===============================
avg_loss = total_loss / n_batches
avg_iou  = total_iou / n_batches
avg_dice = total_dice / n_batches

print("\n" + "="*30)
print("     EVALUATION RESULTS     ")
print("="*30)
print(f"📊 Rata-rata Test Loss: {avg_loss:.4f}")
print(f"🎯 Rata-rata IoU (Jaccard): {avg_iou:.4f}")
print(f"🎲 Rata-rata Dice Score: {avg_dice:.4f}")
print("="*30)
print(f"🖼️ Gambar sampel disimpan di folder: {SAMPLES_DIR.resolve()}")