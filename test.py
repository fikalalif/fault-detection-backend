import torch
import pandas as pd
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
import torch.nn as nn
import torchvision.transforms as T
import matplotlib.pyplot as plt
from dataset_class import FaultDataset
from train import combined_loss, calc_metrics   # pakai fungsi dari train.py

# ===============================
# Config
# ===============================
DATASET_DIR = Path("/home/fikal/fikal/fault_detection/fault-detection-backend/dataset_fault")
MODEL_PATH = Path("models/deeplabv3plus_resnet101_best.pth")
SAMPLES_DIR = Path("samples_test")
SAMPLES_DIR.mkdir(exist_ok=True)

BATCH_SIZE = 8
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ===============================
# Load CSV (pakai validation untuk testing, atau ganti test.csv)
# ===============================
df = pd.read_csv("result_parsing/validation.csv")

def fix_path(path_str):
    filename = path_str.split("\\")[-1]
    return DATASET_DIR / "patched_images" / filename

df["image"] = df["Original Image Patch"].apply(fix_path)
df["mask"]  = df["Binary Mask Image Patch"].apply(fix_path)

transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])

dataset = FaultDataset(df, transform=transform)
loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=False)

# ===============================
# Model
# ===============================
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    in_channels=3,
    classes=1,
    encoder_weights=None
)
model.to(device)

# Load trained weights
checkpoint = torch.load(MODEL_PATH, map_location=device)
if isinstance(checkpoint, dict) and "model" in checkpoint:
    print(f"🔄 Loading FULL checkpoint from {MODEL_PATH}")
    model.load_state_dict(checkpoint["model"])
else:
    print(f"🔄 Loading weights only from {MODEL_PATH}")
    model.load_state_dict(checkpoint)

model.eval()

# ===============================
# Evaluation
# ===============================
bce = nn.BCEWithLogitsLoss()
dice = smp.losses.DiceLoss(mode="binary")

total_loss, total_iou, total_dice = 0, 0, 0
n_batches = len(loader)

with torch.no_grad():
    for i, (imgs, masks) in enumerate(loader):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)

        loss = combined_loss(preds, masks)
        total_loss += loss.item()

        iou, dice_s = calc_metrics(preds, masks)
        total_iou += iou
        total_dice += dice_s

        # Save some sample predictions
        if i < 5:  # simpan 5 batch pertama
            for j in range(len(imgs)):
                img = imgs[j].cpu().permute(1,2,0).numpy()
                mask = masks[j].cpu().squeeze().numpy()
                pred = torch.sigmoid(preds[j]).cpu().squeeze().numpy()
                pred_bin = (pred > 0.5).astype(np.uint8)

                fig, axes = plt.subplots(1, 3, figsize=(12,4))
                axes[0].imshow(img)
                axes[0].set_title("Image")
                axes[1].imshow(mask, cmap="gray")
                axes[1].set_title("Mask (GT)")
                axes[2].imshow(pred_bin, cmap="gray")
                axes[2].set_title("Prediction")
                for ax in axes: ax.axis("off")

                save_path = SAMPLES_DIR / f"sample_{i}_{j}.png"
                plt.savefig(save_path)
                plt.close()

# ===============================
# Print Results
# ===============================
avg_loss = total_loss / n_batches
avg_iou  = total_iou / n_batches
avg_dice = total_dice / n_batches

print(f"\n📊 Test Results")
print(f"Loss: {avg_loss:.4f}")
print(f"IoU : {avg_iou:.4f}")
print(f"Dice: {avg_dice:.4f}")
print(f"✅ Saved some prediction samples to {SAMPLES_DIR}")
