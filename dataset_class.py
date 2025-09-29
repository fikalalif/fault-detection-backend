# nama file: dataset_class.py

import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Definisikan pipeline augmentasi
train_transform = A.Compose(
    [
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)

val_transform = A.Compose(
    [
        A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
        ToTensorV2(),
    ]
)


class FaultDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path, mask_path = row["image"], row["mask"]

        # Buka gambar dan ubah jadi array numpy
        img = np.array(Image.open(img_path).convert("RGB"))
        mask = np.array(Image.open(mask_path).convert("L"))

        # Binarize mask (0/1)
        mask[mask > 0] = 1.0

        if self.transform:
            # Terapkan augmentasi
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
            
            # --- INI PERBAIKANNYA ---
            # Pastikan tipe data mask adalah float sebelum menambahkan dimensi channel
            mask = mask.float()
            
            # Tambahkan channel dimension ke mask (HxW -> 1xHxW)
            mask = mask.unsqueeze(0)

        return img, mask