import pandas as pd
import matplotlib.pyplot as plt
from dataset_class import FaultDataset, JointTransform
from pathlib import Path

DATASET_DIR = Path("/home/fikal/fikal/fault_detection/fault-detection-backend/dataset_fault")
IMG_SIZE = (256, 256)

df = pd.read_csv("result_parsing/train.csv")
def fix_path(path_str):
    filename = path_str.split("\\")[-1]
    return DATASET_DIR / "patched_images" / filename

df["image"] = df["Original Image Patch"].apply(fix_path)
df["mask"]  = df["Binary Mask Image Patch"].apply(fix_path)

dataset = FaultDataset(df, transform=JointTransform(resize=IMG_SIZE))

# Visualisasi 5 sampel augmentasi
for i in range(5):
    img, mask = dataset[i]
    fig, axes = plt.subplots(1, 2, figsize=(6, 3))
    axes[0].imshow(img.permute(1, 2, 0))
    axes[0].set_title("Image")
    axes[1].imshow(mask.squeeze(), cmap="gray")
    axes[1].set_title("Mask")
    for ax in axes: ax.axis("off")
    plt.show()
