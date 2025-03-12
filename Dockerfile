# Use official Python image
FROM python:3.9

# Set working directory inside container
WORKDIR /app

# Copy everything into the container
COPY . /app

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Expose port 8000
EXPOSE 8000

# Run FastAPI app
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
