import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np

class FaultDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self): 
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path, mask_path = row["image"], row["mask"]

        # Buka gambar sebagai array NumPy, ini format yang disukai Albumentations
        try:
            img = np.array(Image.open(img_path).convert("RGB"))
            mask = np.array(Image.open(mask_path).convert("L")) # Grayscale (binary)
        except Exception as e:
            print(f"Error loading image: {img_path} or {mask_path}")
            print(f"Error: {e}")
            # Return data dummy jika error agar training tidak crash
            # Ganti 512 dengan resolusi gambarmu
            return torch.zeros((3, 512, 512)), torch.zeros((1, 512, 512))

        # Binarize mask (0 atau 255)
        # Albumentations lebih mudah jika mask-nya 0 atau 255
        mask[mask > 0] = 255.0

        # Terapkan augmentasi
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        
        # Binarize mask (0/1) SETELAH augmentasi dan ToTensor
        # ToTensorV2 dari Albumentations akan men-scale gambar ke [0, 1]
        # Kita juga men-scale mask ke [0, 1] dan pastikan tipenya float
        mask = (mask / 255.0 > 0.5).float()
        
        # Pastikan mask punya channel dimension
        if mask.dim() == 2:
             mask = mask.unsqueeze(0) # (H, W) -> (1, H, W)

        return img, mask