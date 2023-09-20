from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import io
from io import BytesIO
from PIL import Image
import tensorflow as tf

app = FastAPI()

#MODEL = tf.keras.models.load_model("../model_saved/1")
CLASS_NAMES = ['healthy', 'wilted']

@app.get("/ping")

async def ping():
    return {"Hello world"}

def read_file_as_image(data) -> np.ndarray:
    image = np.array(Image.open(BytesIO(data)))
    return image

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    image = read_file_as_image(await file.read())
    img_batch = np.expand_dims(image, axis=0)
    #predictions = MODEL.predict(img_batch)
    
    #predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    #confidence = round(100 * (np.max(predictions[0])), 2)
    #return {"class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
	uvicorn.run(app, host='localhost', port=8000)