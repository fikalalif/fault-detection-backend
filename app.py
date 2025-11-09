from fastapi import FastAPI, UploadFile, File, HTTPException, Form
from model import predict_mask # <-- Ini dari model.py (deteksi gambar)
# --- FIX: Ganti 'cek_patahan_terdekat' menjadi 'cek_lokasi_patahan' ---
from cek_lokasi_patahan import cek_lokasi_patahan 
from tempfile import NamedTemporaryFile
import shutil
import os
import uuid
from pathlib import Path
from PIL import Image

app = FastAPI()

OUTPUT_DIR = Path("output")
OUTPUT_DIR.mkdir(exist_ok=True)

# ===============================================
# ENDPOINT 1: UNTUK DETEKSI GAMBAR
# ===============================================
@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ["image/jpeg", "image/png"]:
        raise HTTPException(status_code=400, detail="Tipe file tidak valid. Harap upload .jpg atau .png")

    try:
        with NamedTemporaryFile(delete=False, suffix=Path(file.filename).suffix) as tmp:
            shutil.copyfileobj(file.file, tmp)
            tmp_path = tmp.name
        
        try:
            img = Image.open(tmp_path)
            img.verify() 
            img = Image.open(tmp_path) 
            img.load()
        except Exception:
            os.remove(tmp_path) 
            raise HTTPException(status_code=400, detail="File gambar korup atau tidak valid.")

        unique_id = str(uuid.uuid4())
        save_dir = OUTPUT_DIR / unique_id
        save_dir.mkdir(exist_ok=True)

        mask_path, overlay_path, summary = predict_mask(tmp_path, save_dir=str(save_dir))

        if mask_path is None:
            raise HTTPException(status_code=500, detail="Prediksi gambar gagal di server.")

        return {
            "message": "Prediction success",
            "mask_path": mask_path,
            "overlay_path": overlay_path,
            "summary": summary 
        }
    
    finally:
        if 'tmp_path' in locals() and os.path.exists(tmp_path):
            os.remove(tmp_path)

# ===============================================
# ENDPOINT 2: UNTUK DETEKSI LOKASI (FIXED)
# ===============================================
@app.post("/cek_lokasi")
async def cek_lokasi(latitude: float = Form(...), longitude: float = Form(...)):
    try:
        # --- FIX: Panggil nama fungsi yang benar ---
        hasil_cek = cek_lokasi_patahan(latitude, longitude)
        
        # hasil_cek adalah dictionary:
        # {
        #   "status": "Status: ZONA PERINGATAN...",
        #   "nama_patahan": "Sesar Lembang",
        #   "jarak_km": 2.5
        # }
        return hasil_cek

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Gagal memproses lokasi: {str(e)}")

# ===============================================
# Jalankan server ini dengan:
# uvicorn app:app --reload --host 0.0.0.0
# ===============================================