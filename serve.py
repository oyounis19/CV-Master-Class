import uvicorn
import numpy as np
import tensorflow as tf
from utils import labels, number, preprocess_image
from fastapi import FastAPI, File, HTTPException

app = FastAPI()

# Load your trained model
model = tf.keras.models.load_model('model.h5')

@app.get("/")
async def home():
    return {"health": "ok"}

@app.post("/classify")
async def classify(files: list[bytes] = File(...)):
    try:
        # Preprocess the input images
        imgs_array = np.array([preprocess_image(file) for file in files])

        # Make predictions using the model
        preds = np.array([model.predict(img) for img in imgs_array])

        result = {
            f'Image {i}': {
                'number': number[np.argmax(pred)],
                'class': labels[np.argmax(pred)]
            } for i, pred in enumerate(preds)
        }
        result['total'] = sum([number[np.argmax(pred)] for pred in preds])

        return result

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/segment")
async def segment(file: bytes = File(...)):
    try:
        # Preprocess the input image
        img_array = preprocess_image(file)

        # Make predictions using your model
        prediction = model.predict(img_array)
        predicted_label = np.argmax(prediction)
        # You can return the predictions as JSON
        return {'number': number[predicted_label],
                'class': labels[predicted_label]}  # Convert the numpy array to a list and return as JSON
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Run the server
if __name__ == "__main__":
    uvicorn.run("serve:app", host="localhost", port=8000, reload=True)