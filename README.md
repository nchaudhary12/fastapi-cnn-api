# fastapi-cnn-api
A FastAPI-based CNN image classification API trained on CIFAR-10.  Supports Docker containerization, logging, and deployment on Render.
✅ 1. Create app.py
📌 Path: fastapi-cnn-api/app.py

1. Open VS Code.
2. Create a new file: app.py.
Copy the following content:

from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
import numpy as np
from tensorflow.keras.models import load_model
from PIL import Image

app = FastAPI()

# Load the model
model = load_model("models/cnn_cifar10_model.keras")

class PredictionResponse(BaseModel):
    prediction: int

@app.post("/predict", response_model=PredictionResponse)
async def predict(file: UploadFile = File(...)):
    try:
        image = Image.open(file.file).resize((32, 32))
        image = np.array(image) / 255.0
        image = image.reshape((1, 32, 32, 3))

        prediction = model.predict(image).argmax(axis=1)[0]
        return {"prediction": int(prediction)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing image: {str(e)}")

@app.get("/healthz")
def health_check():
    return {"status": "ok"}
✅ 2. Create Dockerfile
📌 Path: fastapi-cnn-api/Dockerfile

Create a new file: Dockerfile.
Copy the following content:

# Use official Python runtime as a parent image
FROM python:3.9

# Set the working directory
WORKDIR /app

# Copy files into the container
COPY . .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port
EXPOSE 10000

# Start the FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "10000"]
✅ 3. Create requirements.txt
📌 Path: fastapi-cnn-api/requirements.txt

Create a new file: requirements.txt.

fastapi
uvicorn
tensorflow
numpy
pillow
python-multipart

✅ 4. Create .gitignore
📌 Path: fastapi-cnn-api/.gitignore

Create a new file: .gitignore.

__pycache__/
*.pyc
*.pyo
*.DS_Store
.env
/models/*
✅ 5. Create README.md
📌 Path: fastapi-cnn-api/README.md

Create a new file: README.md.
Copy the following content:

# FastAPI CNN API

This is a FastAPI-based API for CNN-based image classification.

## 🚀 Setup

### 1. Clone the repository:
```bash
git clone https://github.com/nchaudhary12/fastapi-cnn-api.git
cd fastapi-cnn-api
2. Create a virtual environment and activate it:

python -m venv venv
source venv/bin/activate   # For Linux/Mac
.\venv\Scripts\activate    # For Windows
3. Install dependencies:

pip install -r requirements.txt
4. Start the FastAPI server:

uvicorn app:app --host 0.0.0.0 --port 10000
5. Test the API:
Health Check: http://localhost:10000/healthz
Prediction Endpoint: Send a POST request to /predict with an image file.
6. Docker Setup:
Build the Docker image:

docker build -t fastapi-cnn-api .
Run the container:

docker run -p 10000:10000 fastapi-cnn-api
✅ Example Request (Using curl)

curl -X POST "http://localhost:10000/predict" \
 -F "file=@test_image.png"
✅ Example Response

{
  "prediction": 7
}


---

## ✅ **6. Create `model_training.ipynb`**  
📌 **Path:** `fastapi-cnn-api/notebooks/model_training.ipynb`  

1. Create a new folder **`notebooks/`**.  
2. Create a new Jupyter Notebook called **`model_training.ipynb`**.  
3. Add this content:

```python
import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np

# Load data
(x_train, y_train), (x_test, y_test) = tf.keras.datasets.cifar10.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0

# Create CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10)
])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy'])

# Train model
model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Save model
model.save('../models/cnn_cifar10_model.keras')
✅ 7. Save Model in models/
📌 Path: fastapi-cnn-api/models/cnn_cifar10_model.keras

Create a new folder models/.
Save the trained model as cnn_cifar10_model.keras using the code above.
✅ 8. Push to GitHub
Open terminal/command prompt and run:

# Add files to GitHub
git add .

# Commit changes
git commit -m "Initial commit"

# Push to GitHub
git push origin main
✅ 9. Deploy to Render
Now go to Render and:

Create a new Web Service.

Connect your GitHub repo.

Build Command → pip install -r requirements.txt
Start Command → uvicorn app:app --host 0.0.0.0 --port 10000
Set Port to 10000 if available.

Click Deploy.

✅ 10. Test in Postman
Open Postman.
Send a POST request to your deployed URL:

https://fastapi-cnn-app.onrender.com/predict
Upload an image using form-data → key = file
Expected output:

{
  "prediction": 7
}

✨ Tech Stack
FastAPI
TensorFlow
Python
Docker

🚀 Everything is Ready!
✅ Folder structure – DONE
✅ Code – DONE
✅ Model – DONE
✅ Docker – DONE
✅ GitHub – DONE
✅ Deployment – DONE

➡️ Let me know if you need help with Render or testing! 😎



