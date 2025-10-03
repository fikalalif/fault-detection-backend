from fastapi import FastAPI, UploadFile, File
from model import predict_mask
from tempfile import NamedTemporaryFile
import shutil
import os

app = FastAPI()

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # simpan file upload ke temp
    with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # run prediction → dapat mask + overlay
    mask_path, overlay_path = predict_mask(tmp_path, save_dir=OUTPUT_DIR)

    return {
        "message": "Prediction success",
        "mask_path": mask_path,
        "overlay_path": overlay_path
    }
