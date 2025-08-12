# Use a lightweight Python image (defaults to latest stable, currently 3.13)
FROM python:slim

# Prevent .pyc files and ensure unbuffered output
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# Workdir
WORKDIR /app

# System deps required by LightGBM/XGBoost
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Copy project into the image
# IMPORTANT: our CI have already created the trained model file at the
# same path used by config.path_config.MODEL_OUTPUT_PATH so it's included here.
COPY . .

# Install package (non-editable for a stable image)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir .

# Expose the port your Flask app uses
EXPOSE 5000

# Run the Flask app
CMD ["python", "application.py"]
