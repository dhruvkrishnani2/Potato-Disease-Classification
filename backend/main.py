from fastapi import FastAPI, File, UploadFile
import tensorflow as tf
import numpy as np
from PIL import Image
from io import BytesIO

app = FastAPI()

model = tf.keras.models.load_model("model.h5")

def read_file_as_image(data) -> np.ndarray:
    image = Image.open(BytesIO(data))
    image = image.resize((256, 256))
    return np.array(image)

@app.get("/")
def home():
    return {"message": "API is running"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image / 255.0, axis=0)

    predictions = model.predict(img_batch)
    return {"prediction": str(np.argmax(predictions))}