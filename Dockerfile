# Use official Python image
FROM python:3.10-slim

# Set work directory
WORKDIR /app

# Copy project files
COPY . .

# Install dependencies
RUN pip install --upgrade pip && \
    pip install -r requirements.txt

# Run your script (replace with correct entry point)
CMD ["python", "-m", "src.train"]
