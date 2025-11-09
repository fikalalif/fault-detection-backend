import pandas as pd
import matplotlib.pyplot as plt
from dataset_class import FaultDataset
from pathlib import Path
import albumentations as A
import numpy as np
from PIL import Image
import os

# --- Konfigurasi ---
DATASET_DIR = Path("/home/fikal/fikal/fault_detection/fault-detection-backend/dataset_fault")
patched_images_dir = DATASET_DIR / "patched_images" # Ini sudah benar
IMG_HEIGHT = 512
IMG_WIDTH = 512

# ==========================================================
# ===== BAGIAN DEBUG PATH V3 =====
# ==========================================================
print("="*30)
print("DEBUGGING PATH DATASET V3")
print(f"Target folder gambar: {patched_images_dir}")
print(f"Apakah folder gambar ada? -> {patched_images_dir.exists()}")
print("="*30)

# --- Transformasi ---
train_transform = A.Compose(
    [
        A.Resize(height=IMG_HEIGHT, width=IMG_WIDTH),
        A.Rotate(limit=35, p=1.0),
        A.HorizontalFlip(p=1.0),
        A.VerticalFlip(p=1.0),
        A.RandomBrightnessContrast(p=1.0),
        A.GaussNoise(p=1.0),
    ]
)

# --- Load data ---
csv_path = Path("result_parsing/train.csv")
print(f"Membaca CSV dari: {csv_path.resolve()}")
df = pd.read_csv(csv_path)
print(f"Total baris di CSV: {len(df)}")

# ==========================================================
# ===== BAGIAN DEBUG FILENAME (DENGAN FIX) =====
# ==========================================================
print("="*30)
print("DEBUGGING FILENAME MISMATCH V3")
print("Mengecek 5 baris pertama dari train.csv...")

files_not_found = []

for idx, row in df.head(5).iterrows():
    # 1. Cek file GAMBAR ASLI
    path_from_csv_img = str(row["Original Image Patch"])
    # FIX: Ganti backslash (Windows) ke forward slash (Linux) lalu ambil nama filenya
    filename_csv_img = path_from_csv_img.replace("\\", "/").split("/")[-1]
    img_full_path = patched_images_dir / filename_csv_img
    
    if not img_full_path.exists():
        print(f"❌ GAGAL (Gambar): {img_full_path}")
        files_not_found.append(img_full_path)
    else:
        print(f"✅ OK (Gambar): {filename_csv_img}")

    # 2. Cek file MASK
    path_from_csv_mask = str(row["Binary Mask Image Patch"])
    # FIX: Ganti backslash (Windows) ke forward slash (Linux) lalu ambil nama filenya
    filename_csv_mask = path_from_csv_mask.replace("\\", "/").split("/")[-1]
    mask_full_path = patched_images_dir / filename_csv_mask
    
    if not mask_full_path.exists():
        print(f"❌ GAGAL (Mask): {mask_full_path}")
        files_not_found.append(mask_full_path)
    else:
        print(f"✅ OK (Mask): {filename_csv_mask}")
        
print("="*30)

if files_not_found:
    print("\n!!! DITEMUKAN MASALAH !!!")
    print("Script masih tidak bisa menemukan file di atas.")
    print("Pastikan nama file di atas (setelah 'OK') benar-benar ada di dalam folder 'patched_images' kamu.")
    print("\nScript dihentikan. Perbaiki masalah path/filename di atas.")
    exit() 
else:
    print("\n✅ DEBUG 5 BARIS PERTAMA BERHASIL. Semua file ditemukan.")
    print("Melanjutkan load seluruh dataset...")
# ==========================================================

# --- Fungsi fix_path (SUDAH DIPERBAIKI) ---
def fix_path(path_str):
    if isinstance(path_str, str):
        # FIX: Ganti backslash (Windows) ke forward slash (Linux) lalu ambil nama filenya
        filename = path_str.replace("\\", "/").split("/")[-1]
        full_img_path = patched_images_dir / filename
        if not full_img_path.exists():
            return None # Akan dihapus oleh dropna
        return str(full_img_path)
    return str(path_str)

df["image"] = df["Original Image Patch"].apply(fix_path)
df["mask"]  = df["Binary Mask Image Patch"].apply(fix_path)

initial_count = len(df)
df = df.dropna(subset=["image", "mask"])
if len(df) < initial_count:
    print(f"\nWarning: Dihapus {initial_count - len(df)} baris karena file tidak ditemukan di {patched_images_dir}")

# --- PlottingDataset Class ---
class PlottingDataset(FaultDataset):
     def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path, mask_path = str(row["image"]), str(row["mask"]) 

        try:
            img = np.array(Image.open(img_path).convert("RGB")) 
            mask = np.array(Image.open(mask_path).convert("L"))
        except Exception as e:
            print(f"Error membuka file di index {idx}: {img_path}")
            print(e)
            return np.zeros((IMG_HEIGHT, IMG_WIDTH, 3)), np.zeros((IMG_HEIGHT, IMG_WIDTH))

        mask[mask > 0] = 255.0
        if self.transform:
            augmented = self.transform(image=img, mask=mask)
            img = augmented['image']
            mask = augmented['mask']
        return img, mask

# --- Visualisasi ---
dataset = PlottingDataset(df, transform=train_transform)

if len(dataset) == 0:
    print(f"\nError: Dataset kosong. Ini seharusnya tidak terjadi jika debug di atas berhasil.")
else:
    print(f"\nDataset loaded. Menampilkan 5 sampel augmentasi dari total {len(dataset)} gambar.")
    print("Tutup window plot untuk lanjut ke sampel berikutnya...")

    for i in range(5):
        idx = np.random.randint(0, len(dataset))
        img, mask = dataset[idx]
        
        fig, axes = plt.subplots(1, 2, figsize=(10, 5))
        axes[0].imshow(img) 
        axes[0].set_title(f"Augmented Image (Sample {idx})")
        axes[1].imshow(mask, cmap="gray")
        axes[1].set_title("Augmented Mask")
        for ax in axes: ax.axis("off")
        plt.tight_layout()
        plt.show()