import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from dataset_class import FaultDataset
from tqdm import tqdm
import torchvision.transforms as T

# ===============================
# Config
# ===============================
DATASET_DIR = Path("/home/fikal/fikal/fault_detection/fault-detection-backend/dataset_fault")
EPOCHS = 30
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
val_df["image"] = val_df["Original Image Patch"].apply(fix_path)
val_df["mask"]  = val_df["Binary Mask Image Patch"].apply(fix_path)

# ===============================
# Data Augmentation
# ===============================
train_transform = T.Compose([
    T.Resize((256, 256)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.ColorJitter(brightness=0.2, contrast=0.2),
    T.RandomRotation(15),
    T.ToTensor(),
])

val_transform = T.Compose([
    T.Resize((256, 256)),
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
# Model
# ===============================
model = smp.Unet(
    encoder_name="resnet34", 
    in_channels=3, 
    classes=1, 
    encoder_weights="imagenet"
)
model.to(device)

# Loss & Optimizer
loss_fn = smp.losses.DiceLoss(mode="binary")
optimizer = torch.optim.Adam(model.parameters(), lr=LR)

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
        loss = loss_fn(preds, masks)
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
            loss = loss_fn(preds, masks)
            val_loss += loss.item()

    avg_val_loss = val_loss / len(val_loader)

    print(f"📌 Epoch {epoch}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")

    # Save checkpoint tiap 5 epoch
    if epoch % 5 == 0:
        ckpt_path = CHECKPOINT_DIR / f"unet_resnet34_epoch{epoch}.pth"
        torch.save(model.state_dict(), ckpt_path)
        print(f"💾 Checkpoint saved: {ckpt_path}")

# ===============================
# Save Final Model
# ===============================
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/unet_resnet34_final.pth")
print("✅ Training done, model saved to models/unet_resnet34_final.pth")
