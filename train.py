import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from dataset_class import FaultDataset # Pastikan dataset_class.py sudah di-remake
from tqdm import tqdm
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import csv

# --- Import BARU untuk Augmentasi ---
import albumentations as A
from albumentations.pytorch import ToTensorV2
# --- Hapus: import torchvision.transforms as T ---

# ===============================
# Config
# ===============================
DATASET_DIR = Path("/home/fikal/fikal/fault_detection/fault-detection-backend/dataset_fault")
EPOCHS = 100
BATCH_SIZE = 4 # Sesuaikan jika VRAM 4GB tidak cukup, mungkin turun ke 4
LR = 1e-4
PATIENCE = 7   # early stopping patience
CHECKPOINT_DIR = Path("checkpoints")
SAMPLES_DIR = Path("samples")
MODEL_DIR = Path("models")
LOG_PATH = Path("training_log.csv")
CHECKPOINT_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# --- BARU: Tentukan Ukuran Gambar ---
IMG_HEIGHT = 512
IMG_WIDTH = 512

# path ke checkpoint (ubah sesuai file kamu jika ingin resume)
RESUME_FROM = CHECKPOINT_DIR / "deeplabv3plus_resnet34_epoch25.pth"

# ===============================
# BARU: Definisikan Augmentasi (Albumentations)
# ===============================
train_transform = A.Compose(
    [
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Rotate(limit=35, p=0.7),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomBrightnessContrast(p=0.4),
        A.GaussNoise(p=0.2),
        A.Normalize(
            mean=[0.0, 0.0, 0.0],
            std=[1.0, 1.0, 1.0],
            max_pixel_value=255.0,
        ),
        ToTensorV2(), # Konversi ke Tensor
    ]
)

val_transform = A.Compose(
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

# ===============================
# Load CSV (Dengan Fungsi fix_path yang BENAR)
# ===============================
train_df = pd.read_csv("result_parsing/train.csv")
val_df   = pd.read_csv("result_parsing/validation.csv")
patched_images_dir = DATASET_DIR / "patched_images" # Definisikan path gambar di sini

def fix_path(path_str):
    if isinstance(path_str, str):
        # FIX: Ganti backslash (Windows) ke forward slash (Linux) lalu ambil nama filenya
        filename = path_str.replace("\\", "/").split("/")[-1]
        
        full_img_path = patched_images_dir / filename
        if not full_img_path.exists():
            # print(f"Warning: File not found {full_img_path}") # Aktifkan untuk debug
            return None # Akan dihapus oleh dropna
        return str(full_img_path)
    return str(path_str) # Kembalikan jika sudah benar

print("Memproses path CSV...")
train_df["image"] = train_df["Original Image Patch"].apply(fix_path)
train_df["mask"]  = train_df["Binary Mask Image Patch"].apply(fix_path)
val_df["image"]   = val_df["Original Image Patch"].apply(fix_path)
val_df["mask"]    = val_df["Binary Mask Image Patch"].apply(fix_path)

# --- BARU: Hapus baris yang filenya tidak ditemukan ---
train_df = train_df.dropna(subset=["image", "mask"])
val_df = val_df.dropna(subset=["image", "mask"])

print(f"Total Training data loaded: {len(train_df)} samples")
print(f"Total Validation data loaded: {len(val_df)} samples")

if len(train_df) == 0 or len(val_df) == 0:
    print("!!! ERROR: Dataset kosong. Pastikan path dan CSV benar.")
    exit()

# ===============================
# Dataset & Dataloader
# ===============================
print("Loading Datasets...")
train_dataset = FaultDataset(df=train_df, transform=train_transform) # Ganti transform
val_dataset   = FaultDataset(df=val_df, transform=val_transform)     # Ganti transform

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4, # Sesuaikan dengan CPU-mu
    pin_memory=True,
    shuffle=True
)
val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    num_workers=4,
    pin_memory=True,
    shuffle=False
)
print("Datasets Loaded.")

# ===============================
# Model, Loss, Optimizer
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = smp.DeepLabV3Plus(
    encoder_name="resnet34",  # <--- GANTI INI
    in_channels=3,
    classes=1,
    encoder_weights="imagenet" 
)
model.to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, "min", patience=3, factor=0.1)

# Loss function
dice_loss = smp.losses.DiceLoss(mode="binary")
bce_loss = nn.BCEWithLogitsLoss()
def combined_loss(preds, masks):
    return 0.5 * dice_loss(preds, masks) + 0.5 * bce_loss(preds, masks)

# Metrics
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
# Setup Logging
# ===============================
with open(LOG_PATH, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["epoch", "train_loss", "val_loss", "iou", "dice"])

# ===============================
# Training Loop
# ===============================
best_val_loss = float("inf")
no_improve = 0
start_epoch = 1

# --- (Opsional) Resume training ---
if Path(RESUME_FROM).exists():
    print(f"Resuming from {RESUME_FROM}")
    checkpoint = torch.load(RESUME_FROM)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    start_epoch = checkpoint["epoch"] + 1
    best_val_loss = checkpoint.get("best_val_loss", float("inf"))


print("Starting Training...")
for epoch in range(start_epoch, EPOCHS + 1):
    model.train()
    train_loss = 0

    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch} [Train]"):
        imgs, masks = imgs.to(device), masks.to(device)
        
        optimizer.zero_grad()
        preds = model(imgs)
        loss = combined_loss(preds, masks)
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
    
    avg_train_loss = train_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss = 0
    val_iou = 0
    val_dice = 0

    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch} [Val]"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            
            val_loss += combined_loss(preds, masks).item()
            iou, dice_s = calc_metrics(preds, masks)
            val_iou += iou
            val_dice += dice_s

    avg_val_loss = val_loss / len(val_loader)
    avg_val_iou = val_iou / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)

    scheduler.step(avg_val_loss)

    print(f"📌 Epoch {epoch}/{EPOCHS} - "
          f"Train Loss: {avg_train_loss:.4f} | "
          f"Val Loss: {avg_val_loss:.4f} | "
          f"IoU: {avg_val_iou:.4f} | Dice: {avg_val_dice:.4f}")

    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([epoch, avg_train_loss, avg_val_loss, avg_val_iou, avg_val_dice])

    # ---- Early Stopping ----
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        no_improve = 0
        torch.save(model.state_dict(), MODEL_DIR / "deeplabv3plus_resnet34_best.pth")
        print(f"💾 Best model saved (val_loss={best_val_loss:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print(f"⛔ Early stopping triggered after {PATIENCE} epochs with no improvement.")
            break

    # ---- Save checkpoint tiap 5 epoch ----
    if epoch % 5 == 0:
        checkpoint_path = CHECKPOINT_DIR / f"deeplabv3plus_resnet34_epoch{epoch}.pth"
        chkpt = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "best_val_loss": best_val_loss
        }
        torch.save(chkpt, checkpoint_path)
        print(f"📦 Checkpoint saved to {checkpoint_path}")

print("Training finished.")