from fastapi import FastAPI, File, UploadFile
import uvicorn
import numpy as np
import io
import os
from io import BytesIO
from PIL import Image
import tensorflow as tf
### use os to list all directories in the model_saved folder
# all_files = os.listdir("/Users/bhargobdeka/Desktop/Projects/houseplant-healthy/model_saved/1")
# print(all_files)
if os.path.exists("/Users/bhargobdeka/Desktop/Projects/houseplant-healthy/model_saved/1"):
    MODEL = tf.keras.models.load_model("/Users/bhargobdeka/Desktop/Projects/houseplant-healthy/model_saved/1")
else:
    print("Model file does not exist at the expected path.")

# pip install numpy~=1.19.5 fixed the issue to load the above model

app = FastAPI()


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
    img_batch = np.expand_dims(image, 0)
    predictions = MODEL.predict(img_batch)
    
    predicted_class = CLASS_NAMES[np.argmax(predictions[0])]
    confidence = round(100 * (np.max(predictions[0])), 2)
    return {"class": predicted_class, "confidence": confidence}

if __name__ == "__main__":
	uvicorn.run(app, host='localhost', port=8080)