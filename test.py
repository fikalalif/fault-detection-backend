import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import DataLoader
import segmentation_models_pytorch as smp
from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from dataset_class import FaultDataset
from tqdm import tqdm

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
# Load CSV
# ===============================
val_df = pd.read_csv("result_parsing/validation.csv")

def fix_path(path_str):
    filename = path_str.split("\\")[-1]
    return DATASET_DIR / "patched_images" / filename

val_df["image"] = val_df["Original Image Patch"].apply(fix_path)
val_df["mask"]  = val_df["Binary Mask Image Patch"].apply(fix_path)

# ===============================
# Transform
# ===============================
val_transform = T.Compose([
    T.Resize((512, 512)),
    T.ToTensor(),
])

val_dataset = FaultDataset(val_df, transform=val_transform)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)

# ===============================
# Model
# ===============================
model = smp.DeepLabV3Plus(
    encoder_name="resnet101",
    in_channels=3,
    classes=1,
    encoder_weights=None  # best model sudah pretrained
)
model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
model.to(device)
model.eval()

# ===============================
# Metrics (manual IoU & Dice)
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
# Visualization
# ===============================
def visualize_prediction(img, mask, pred, idx):
    img_np  = img.permute(1, 2, 0).cpu().numpy()
    mask_np = mask.squeeze().cpu().numpy()
    pred_np = torch.sigmoid(pred).squeeze().cpu().numpy()
    pred_bin = (pred_np > 0.5).astype(np.uint8)

    fig, axs = plt.subplots(1, 4, figsize=(16, 4))
    axs[0].imshow(img_np); axs[0].set_title("Input")
    axs[1].imshow(mask_np, cmap="gray"); axs[1].set_title("Ground Truth")
    axs[2].imshow(pred_np, cmap="magma"); axs[2].set_title("Pred (Prob)")
    axs[3].imshow(img_np); axs[3].imshow(pred_bin, cmap="Reds", alpha=0.4); axs[3].set_title("Overlay")
    for ax in axs: ax.axis("off")

    save_path = SAMPLES_DIR / f"sample_{idx}.png"
    plt.savefig(save_path)
    plt.close()

# ===============================
# Testing
# ===============================
total_iou, total_dice, count = 0, 0, 0

with torch.no_grad():
    for idx, (imgs, masks) in enumerate(tqdm(val_loader, desc="Testing")):
        imgs, masks = imgs.to(device), masks.to(device)
        preds = model(imgs)

        iou, dice = calc_metrics(preds, masks)
        total_iou += iou
        total_dice += dice
        count += 1

        # simpan 2 sample pertama tiap batch
        for i in range(min(2, imgs.size(0))):
            visualize_prediction(imgs[i].cpu(), masks[i].cpu(), preds[i].cpu(), f"{idx}_{i}")

avg_iou = total_iou / count
avg_dice = total_dice / count

print(f"✅ Test done - IoU: {avg_iou:.4f} | Dice: {avg_dice:.4f}")
