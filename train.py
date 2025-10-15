import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from dataset_class import FaultDataset
from tqdm import tqdm
import torchvision.transforms as T
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy as np
import csv

# ===============================
# Config
# ===============================
DATASET_DIR = Path("/home/fikal/fikal/fault_detection/fault-detection-backend/dataset_fault")
EPOCHS = 100
BATCH_SIZE = 8
LR = 1e-4
PATIENCE = 7   # early stopping patience
CHECKPOINT_DIR = Path("checkpoints")
SAMPLES_DIR = Path("samples")
MODEL_DIR = Path("models")
LOG_PATH = Path("training_log.csv")
CHECKPOINT_DIR.mkdir(exist_ok=True)
SAMPLES_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)

# path ke checkpoint (ubah sesuai file kamu)
RESUME_FROM = CHECKPOINT_DIR / "deeplabv3plus_resnet101_epoch40.pth"

# ===============================
# Load CSV
# ===============================
train_df = pd.read_csv("result_parsing/train.csv")
val_df   = pd.read_csv("result_parsing/validation.csv")

def fix_path(path_str):
    filename = path_str.split("\\")[-1]
    return DATASET_DIR / "patched_images" / filename

train_df["image"] = train_df["Original Image Patch"].apply(fix_path)
train_df["mask"]  = train_df["Binary Mask Image Patch"].apply(fix_path)
val_df["image"]   = val_df["Original Image Patch"].apply(fix_path)
val_df["mask"]    = val_df["Binary Mask Image Patch"].apply(fix_path)

# ===============================
# Data Augmentation
# ===============================
train_transform = T.Compose([
    T.Resize((512, 512)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomRotation(20),
    T.ToTensor(),
])

val_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])

train_dataset = FaultDataset(train_df, transform=train_transform)
val_dataset   = FaultDataset(val_df, transform=val_transform)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ===============================
# Device
# ===============================
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Model (DeepLabv3+)
# ===============================
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    in_channels=3,
    classes=1,
    encoder_weights="imagenet"
)
model.to(device)

# ===============================
# Loss & Optimizer
# ===============================
bce = nn.BCEWithLogitsLoss()
dice = smp.losses.DiceLoss(mode="binary")

def combined_loss(pred, target):
    return bce(pred, target) + dice(pred, target)

optimizer = torch.optim.Adam(model.parameters(), lr=LR)
scaler = torch.amp.GradScaler("cuda")

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=3
)

# ===============================
# Metrics
# ===============================
def calc_metrics(pred, target, threshold=0.5):
    if target.dim() == 3:
        target = target.unsqueeze(1)

    prob = torch.sigmoid(pred)
    pred_bin = (prob > threshold).float()

    pred_flat = pred_bin.view(-1)
    target_flat = target.view(-1)

    intersection = (pred_flat * target_flat).sum()
    union = pred_flat.sum() + target_flat.sum() - intersection
    dice = (2.0 * intersection) / (pred_flat.sum() + target_flat.sum() + 1e-7)
    iou = intersection / (union + 1e-7)

    return float(iou), float(dice)

# ===============================
# Resume Checkpoint
# ===============================
start_epoch = 1
best_val_loss = float("inf")
no_improve = 0

if RESUME_FROM.exists():
    checkpoint = torch.load(RESUME_FROM, map_location=device)

    if isinstance(checkpoint, dict) and "model" in checkpoint:
        # ✅ format baru
        print(f"🔄 Loading FULL checkpoint from {RESUME_FROM}")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scaler.load_state_dict(checkpoint["scaler"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        start_epoch = checkpoint["epoch"] + 1
        best_val_loss = checkpoint["best_val_loss"]
    else:
        # ✅ format lama (hanya bobot model)
        print(f"🔄 Loading model weights only from {RESUME_FROM}")
        model.load_state_dict(checkpoint)
        start_epoch = 21   # <<-- karena lu udah stop di epoch 20
        best_val_loss = float("inf")
else:
    print("⚡ Training from scratch")


# ===============================
# Training Loop
# ===============================
if not LOG_PATH.exists():
    with open(LOG_PATH, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["epoch", "train_loss", "val_loss", "IoU", "Dice"])

for epoch in range(start_epoch, EPOCHS+1):
    # ---- Train ----
    model.train()
    train_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
        imgs, masks = imgs.to(device), masks.to(device)
        optimizer.zero_grad()

        with torch.amp.autocast("cuda"):
            preds = model(imgs)
            loss = combined_loss(preds, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        train_loss += loss.item()

    avg_train_loss = train_loss / len(train_loader)

    # ---- Validation ----
    model.eval()
    val_loss, val_iou, val_dice = 0, 0, 0
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = combined_loss(preds, masks)
            val_loss += loss.item()

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
        torch.save(model.state_dict(), MODEL_DIR / "deeplabv3plus_resnet101_best.pth")
        print(f"💾 Best model saved (val_loss={best_val_loss:.4f})")
    else:
        no_improve += 1
        if no_improve >= PATIENCE:
            print("⛔ Early stopping triggered")
            break

    # ---- Save checkpoint tiap 5 epoch ----
    if epoch % 5 == 0:
        ckpt_path = CHECKPOINT_DIR / f"deeplabv3plus_resnet101_epoch{epoch}.pth"
        torch.save({
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_val_loss": best_val_loss
        }, ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")