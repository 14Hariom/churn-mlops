FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

# libgomp is needed by xgboost / scikit-learn on the slim image
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /code

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY app/ app/
COPY data/ data/

# Train the model while building the image, so the model always matches the
# library versions installed in the image.
RUN python -m src.train --no-mlflow

EXPOSE 8000

# Render (and most hosts) provide the port in $PORT; default to 8000
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]
