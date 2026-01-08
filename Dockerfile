# Base Image
FROM python:3.9-slim
# Set the working directory
WORKDIR /app
# Copy requirements file 
COPY requirements.txt .
# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt
# Copy the application code
COPY api/ ./api/
# Create models directory (model will be mounted at runtime in production)
RUN mkdir -p ./models
# Expose the port
EXPOSE 8000
# Start Command
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]