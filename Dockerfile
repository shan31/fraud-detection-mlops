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
# Copy the trained model
COPY models/ ./models/
# Expose the port
EXPOSE 8000
# Start Command
CMD ["uvicorn", "api.app:app", "--host", "0.0.0.0", "--port", "8000"]