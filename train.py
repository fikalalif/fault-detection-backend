# nama file: train.py

import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from dataset_class import FaultDataset, train_transform, val_transform # Import transform dari dataset_class
from tqdm import tqdm
import torch.nn.functional as F

# --- FUNGSI UNTUK METRIK ---
def dice_score(pred, target, smooth=1.):
    pred = torch.sigmoid(pred)
    pred = (pred > 0.5).float()
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    dice = (2. * intersection + smooth) / (union + smooth)
    return dice.item()

# --- SETUP ---
DATASET_DIR = Path("/home/fikal/fikal/fault_detection/fault-detection-backend/dataset_fault")
PATCH_DIR = DATASET_DIR / "patched_images"

# Load csv hasil parse
train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("validation.csv")

# Fix paths
def fix_path(path_str):
    filename = path_str.split("\\")[-1]
    return PATCH_DIR / filename

train_df["image"] = train_df["Original Image Patch"].apply(fix_path)
train_df["mask"]  = train_df["Binary Mask Image Patch"].apply(fix_path)
val_df["image"] = val_df["Original Image Patch"].apply(fix_path)
val_df["mask"]  = val_df["Binary Mask Image Patch"].apply(fix_path)

# --- MODIFIED: Gunakan transform yang sudah kita buat ---
train_dataset = FaultDataset(train_df, transform=train_transform)
val_dataset = FaultDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4) # Naikkan batch size jika VRAM cukup
val_loader   = DataLoader(val_dataset, batch_size=16, num_workers=4)

# Device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Model
model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, encoder_weights="imagenet")
model.to(device)

# Loss & Optimizer
# Menggabungkan BCE Loss dan Dice Loss untuk hasil yang lebih stabil
loss_fn = smp.losses.DiceLoss(mode="binary")
bce_loss = torch.nn.BCEWithLogitsLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# --- MODIFIED: Training loop yang lebih baik ---
EPOCHS = 50 # Tingkatkan jumlah epoch
best_val_loss = float('inf') # Simpan loss terbaik
patience = 5 # Jumlah epoch untuk menunggu perbaikan sebelum berhenti
epochs_no_improve = 0 # Counter untuk early stopping

# Membuat folder models jika belum ada
Path("models").mkdir(exist_ok=True)
model_save_path = "models/unet_resnet34_best_model.pth"

for epoch in range(EPOCHS):
    print(f"\n--- Epoch {epoch+1}/{EPOCHS} ---")
    
    # --- Training Phase ---
    model.train()
    train_loss = 0
    train_dice = 0
    
    for imgs, masks in tqdm(train_loader, desc="Training"):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        
        # Kalkulasi loss gabungan
        loss_d = loss_fn(preds, masks)
        loss_b = bce_loss(preds, masks)
        loss = loss_d + loss_b # Kombinasi loss
        
        loss.backward()
        optimizer.step()
        
        train_loss += loss.item()
        train_dice += dice_score(preds, masks)
    
    avg_train_loss = train_loss / len(train_loader)
    avg_train_dice = train_dice / len(train_loader)
    print(f"Train Loss: {avg_train_loss:.4f} | Train Dice Score: {avg_train_dice:.4f}")

    # --- Validation Phase ---
    model.eval()
    val_loss = 0
    val_dice = 0
    
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc="Validation"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            
            # Kalkulasi loss gabungan
            loss_d = loss_fn(preds, masks)
            loss_b = bce_loss(preds, masks)
            loss = loss_d + loss_b
            
            val_loss += loss.item()
            val_dice += dice_score(preds, masks)

    avg_val_loss = val_loss / len(val_loader)
    avg_val_dice = val_dice / len(val_loader)
    print(f"Validation Loss: {avg_val_loss:.4f} | Validation Dice Score: {avg_val_dice:.4f}")

    # --- Early Stopping & Save Best Model ---
    if avg_val_loss < best_val_loss:
        print(f"✅ Validation loss improved ({best_val_loss:.4f} --> {avg_val_loss:.4f}). Saving model...")
        best_val_loss = avg_val_loss
        torch.save(model.state_dict(), model_save_path)
        epochs_no_improve = 0 # Reset counter
    else:
        epochs_no_improve += 1
        print(f"⚠️ Validation loss did not improve. Counter: {epochs_no_improve}/{patience}")

    if epochs_no_improve >= patience:
        print(f"Stopping early as validation loss did not improve for {patience} epochs.")
        break

print(f"\n✅ Training finished. Best model saved to {model_save_path}")