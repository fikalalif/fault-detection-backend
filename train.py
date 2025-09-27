import pandas as pd
from pathlib import Path
from torch.utils.data import DataLoader
import torch
import segmentation_models_pytorch as smp
from dataset_class import FaultDataset
from tqdm import tqdm

# dataset paths
DATASET_DIR = Path("/home/fikal/fikal/fault_detection/fault-detection-backend/dataset_fault")

# load csv hasil parse
train_df = pd.read_csv("train.csv")
val_df   = pd.read_csv("validation.csv")

# fix paths
def fix_path(path_str):
    filename = path_str.split("\\")[-1]
    return DATASET_DIR / "patched_images" / filename

train_df["image"] = train_df["Original Image Patch"].apply(fix_path)
train_df["mask"]  = train_df["Binary Mask Image Patch"].apply(fix_path)
val_df["image"] = val_df["Original Image Patch"].apply(fix_path)
val_df["mask"]  = val_df["Binary Mask Image Patch"].apply(fix_path)

# dataset + dataloader
train_dataset = FaultDataset(train_df)
val_dataset = FaultDataset(val_df)

train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
val_loader   = DataLoader(val_dataset, batch_size=8)

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# model: U-Net with ResNet34 backbone
model = smp.Unet(encoder_name="resnet34", in_channels=3, classes=1, encoder_weights="imagenet")
model.to(device)

# loss & optimizer
loss_fn = smp.losses.DiceLoss(mode="binary")
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# training loop
EPOCHS = 5
for epoch in range(EPOCHS):
    model.train()
    train_loss = 0
    for imgs, masks in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        imgs, masks = imgs.to(device), masks.to(device)

        optimizer.zero_grad()
        preds = model(imgs)
        loss = loss_fn(preds, masks)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()

    print(f"Epoch {epoch+1} - Train Loss: {train_loss/len(train_loader):.4f}")

# save model
Path("models").mkdir(exist_ok=True)
torch.save(model.state_dict(), "models/unet_resnet34.pth")
print("✅ Model saved to models/unet_resnet34.pth")
