import cv2
import numpy as np
from PIL import Image
import os

def overlay_mask(image, mask, color=(255, 0, 0), alpha=0.5):
    overlay = np.array(image).copy()
    mask_rgb = np.zeros_like(overlay)
    mask_rgb[mask == 1] = color
    return cv2.addWeighted(overlay, 1 - alpha, mask_rgb, alpha, 0)

def predict_mask(image_path, save_dir="output"):
    os.makedirs(save_dir, exist_ok=True)

    # buka gambar asli
    img = Image.open(image_path).convert("RGB")
    img_np = np.array(img)

    # dummy: mask dari grayscale (nanti ganti pakai model asli)
    gray = cv2.cvtColor(img_np, cv2.COLOR_RGB2GRAY)
    _, mask = cv2.threshold(gray, 127, 1, cv2.THRESH_BINARY)

    # simpan mask
    mask_img = Image.fromarray((mask * 255).astype(np.uint8))
    mask_path = os.path.join(save_dir, "mask.png")
    mask_img.save(mask_path)

    # simpan overlay
    overlay = overlay_mask(img, mask)
    overlay_img = Image.fromarray(overlay)
    overlay_path = os.path.join(save_dir, "overlay.png")
    overlay_img.save(overlay_path)

    return mask_path, overlay_path
