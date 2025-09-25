import pandas as pd
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt

# dataset path
DATASET_DIR = Path("/home/fikal/fikal/fault_detection/fault-detection-backend/dataset_fault")
PATCH_DIR = DATASET_DIR / "patched_images"

# load csv
train_df = pd.read_csv("train.csv")

# fungsi convert path
def fix_path(path_str):
    # ambil nama file aja
    filename = path_str.split("\\")[-1]
    return PATCH_DIR / filename

# convert kolom
train_df["image"] = train_df["Original Image Patch"].apply(fix_path)
train_df["mask"]  = train_df["Binary Mask Image Patch"].apply(fix_path)

print(train_df.head())

# cek contoh gambar
img_path = train_df.iloc[0]["image"]
mask_path = train_df.iloc[0]["mask"]

print("Image:", img_path)
print("Mask:", mask_path)

# plot
img = Image.open(img_path)
mask = Image.open(mask_path)

plt.subplot(1,2,1)
plt.imshow(img)
plt.title("Original Patch")

plt.subplot(1,2,2)
plt.imshow(mask, cmap="gray")
plt.title("Mask Patch")
plt.show()
