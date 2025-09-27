from fastapi import FastAPI, UploadFile, File
from model import predict_mask
from tempfile import NamedTemporaryFile
import shutil

app = FastAPI()

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    # save temp file
    with NamedTemporaryFile(delete=False, suffix=".png") as tmp:
        shutil.copyfileobj(file.file, tmp)
        tmp_path = tmp.name

    # run prediction
    mask = predict_mask(tmp_path, save_path="output/result.png")

    return {"message": "Prediction success", "output_path": "output/result.png"}
