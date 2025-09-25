import pandas as pd
from pathlib import Path

# path ke dataset
DATASET_DIR = Path("/home/fikal/fikal/fault_detection/fault-detection-backend/dataset_fault")

# baca .tab files
train_df = pd.read_csv(DATASET_DIR / "train.tab", sep="\t")
val_df   = pd.read_csv(DATASET_DIR / "validation.tab", sep="\t")
test_df  = pd.read_csv(DATASET_DIR / "test.tab", sep="\t")

print("🔹 Train set:", train_df.shape)
print("🔹 Validation set:", val_df.shape)
print("🔹 Test set:", test_df.shape)

# lihat contoh 5 baris pertama
print("\nContoh train data:")
print(train_df.head())

# save ke CSV biar gampang dibaca lagi
train_df.to_csv("train.csv", index=False)
val_df.to_csv("validation.csv", index=False)
test_df.to_csv("test.csv", index=False)

print("\n✅ Data berhasil diparse dan disimpan ke CSV")
