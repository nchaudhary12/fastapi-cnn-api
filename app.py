import logging
from fastapi import FastAPI, File, UploadFile, HTTPException
from tensorflow import keras
import numpy as np
from PIL import Image, UnidentifiedImageError
import uvicorn
import io
import os
from datetime import datetime

# Initialize logging
logging.basicConfig(
    filename="app.log",  # Logs will be saved in app.log
    level=logging.INFO,  
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Define class labels (same order as training)
LABELS = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# Dynamically set model path (Works for both Local & Docker)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "cnn_cifar10_model.keras")

# Load the trained model safely
try:
    model = keras.models.load_model(MODEL_PATH)
    logging.info(f"Model loaded successfully from: {MODEL_PATH}")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    raise RuntimeError("Model file not found or corrupted!")

# Initialize FastAPI app
app = FastAPI(
    title="CNN Image Classifier API",
    description="A simple API for classifying CIFAR-10 images using a trained CNN model.",
    version="1.0"
)

# Homepage (Root Endpoint)
@app.get("/", summary="Welcome Page")
def read_root():
    return {
        "message": "Welcome to the CNN Image Classifier API!",
        "docs_url": "/docs",
        "usage": "Use /predict to upload an image for classification."
    }

# Function to preprocess uploaded image
def preprocess_image(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")  # Convert to RGB to avoid mode issues
    image = image.resize((32, 32))  # Resize to match CIFAR-10 dimensions
    image = np.array(image) / 255.0  # Normalize pixel values
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

# API endpoint for predictions with logging & error handling
@app.post("/predict", summary="Predict Image Class", description="Upload an image to classify it into one of the CIFAR-10 categories.")
async def predict(file: UploadFile = File(...)):
    try:
        logging.info(f"Received request: {file.filename} at {datetime.now()}")

        # Read file into bytes and open as an image
        image_bytes = await file.read()
        image = Image.open(io.BytesIO(image_bytes))
        
        # Preprocess image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = model.predict(processed_image)
        predicted_class = LABELS[np.argmax(prediction)]

        logging.info(f"Prediction successful: {file.filename} -> {predicted_class}")

        return {
            "filename": file.filename,
            "predicted_class": predicted_class
        }

    except UnidentifiedImageError:
        logging.error(f"Invalid image file: {file.filename}")
        raise HTTPException(status_code=400, detail="Invalid image file. Please upload a valid image format.")
    
    except Exception as e:
        logging.error(f"Unexpected error: {e}")
        raise HTTPException(status_code=500, detail="An internal server error occurred.")

# Run the FastAPI server
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)





