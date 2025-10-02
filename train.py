import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from dataset_class import FaultDataset
from tqdm import tqdm
import torchvision.transforms as T
import torch.nn as nn

# ===============================
# Config
# ===============================
DATASET_DIR = Path("/home/fikal/fikal/fault_detection/fault-detection-backend/dataset_fault")
EPOCHS = 50
BATCH_SIZE = 8
LR = 1e-4
CHECKPOINT_DIR = Path("checkpoints")
CHECKPOINT_DIR.mkdir(exist_ok=True)

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

# Scheduler tanpa verbose (compat versi lama)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer, factor=0.5, patience=3
)

# ===============================
# Training Loop
# ===============================
for epoch in range(1, EPOCHS+1):
    # ---- Train ----
    model.train()
    train_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS} [Train]"):
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
    with torch.no_grad():
        for imgs, masks in tqdm(val_loader, desc=f"Epoch {epoch}/{EPOCHS} [Val]"):
            imgs, masks = imgs.to(device), masks.to(device)
            preds = model(imgs)
            loss = combined_loss(preds, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    # Scheduler step + manual logging
    old_lr = optimizer.param_groups[0]['lr']
    scheduler.step(avg_val_loss)
    new_lr = optimizer.param_groups[0]['lr']
    if new_lr < old_lr:
        print(f"⚡ Learning rate reduced: {old_lr:.6f} → {new_lr:.6f}")

    print(f"📌 Epoch {epoch}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save checkpoint tiap 5 epoch
    if epoch % 5 == 0:
        ckpt_path = CHECKPOINT_DIR / f"deeplabv3plus_resnet101_epoch{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")

# ===============================
# Save Final Model
# ===============================
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/deeplabv3plus_resnet101_final.pth")
print("✅ Training done, model saved to models/deeplabv3plus_resnet101_final.pth")
