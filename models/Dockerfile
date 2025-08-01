# Step 1: Use Python 3.10 as base
FROM python:3.10-slim

# Step 2: Set the working directory in the container
WORKDIR /app

# Step 3: Copy dependency list and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Step 4: Copy source code and model
COPY src/ src/
COPY models/ models/

# Step 5: Set the default command to run prediction
CMD ["python", "src/predict.py"]
