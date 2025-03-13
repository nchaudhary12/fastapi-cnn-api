# fastapi-cnn-api
A FastAPI-based CNN image classification API trained on CIFAR-10.  Supports Docker containerization, logging, and deployment on Render.

#Clone the repository
git clone https://github.com/nchaudhary12/fastapi-cnn-api.git  
cd fastapi-cnn-api

#Install dependencies
pip install -r requirements.txt

#Run the API
uvicorn app:app --host 0.0.0.0 --port 10000

#Test the API
Open Postman
Send a POST request to:
http://localhost:10000/predict/
Set form-data → Key: file, Value: (Upload an image)

#Docker Setup
Build Docker Image
docker build -t fastapi-cnn-api .
Run Docker Container
docker run -p 10000:10000 fastapi-cnn-api

#Deployment on Render
1. Create a Web Service on Render.
2. Build Command → pip install -r requirements.txt
3. Start Command → uvicorn app:app --host 0.0.0.0 --port 10000
https://<your-app-name>.onrender.com/predict/

#Example Response

{
  "prediction": "cat",
  "confidence": 0.95
}

#Project Structure 
fastapi-cnn-api/
├── app.py                          # FastAPI code (backend logic)
├── Dockerfile                      # Docker setup
├── requirements.txt                # List of dependencies
├── models/                         
│   └── cnn_cifar10_model.keras     # Trained model file
├── README.md                       # Project documentation
├── .gitignore                      # Files to ignore in Git
└── notebooks/                      
    └── model_training.ipynb        # Model training code

#Tech Stack
FastAPI
TensorFlow
Python
Docker
Uvicorn

