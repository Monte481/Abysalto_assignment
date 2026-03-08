FROM python:3.12-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir -r requirements.txt

COPY backend /app/backend
COPY dummy_docs /app/dummy_docs

WORKDIR /app/backend

EXPOSE 5000

CMD ["python", "app.py"]