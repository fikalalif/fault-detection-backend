import torch
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image

class FaultDataset(Dataset):
    def __init__(self, df, transform=None):
        self.df = df
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path, mask_path = row["image"], row["mask"]

        # open images
        img = Image.open(img_path).convert("RGB")
        mask = Image.open(mask_path).convert("L")  # grayscale (binary)

        # to tensor
        img = transforms.ToTensor()(img)
        mask = transforms.ToTensor()(mask)

        # binarize mask (0/1)
        mask = (mask > 0.5).float()

        return img, mask