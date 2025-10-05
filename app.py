from fastapi import FastAPI, UploadFile, File
from model import predict_mask
from tempfile import NamedTemporaryFile
import shutil
import os
import uuid

app = FastAPI()

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # simpan file upload ke temp
    with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # bikin folder unik untuk hasil
    unique_id = str(uuid.uuid4())
    save_dir = os.path.join(OUTPUT_DIR, unique_id)
    os.makedirs(save_dir, exist_ok=True)

    # run prediction → dapat mask + overlay
    mask_path, overlay_path = predict_mask(tmp_path, save_dir=save_dir)

    return {
        "message": "Prediction success",
        "mask_path": mask_path,
        "overlay_path": overlay_path
    }