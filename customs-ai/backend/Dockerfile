# Faza 2 (server) uchun — air-gapped registry'dan build qilinadi.
# Faza 1 (laptop) Windows service sifatida ishlaydi (Docker emas).
FROM python:3.11-slim

WORKDIR /app

# Offline build: wheel'lar kontekstda (deploy/offline-wheels/wheels) bo'lishi mumkin.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY config ./config

ENV CUSTOMS_HOST=0.0.0.0 \
    CUSTOMS_PORT=8000 \
    CUSTOMS_USE_MOCKS=false

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
