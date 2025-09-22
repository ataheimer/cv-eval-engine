# ---------- base ----------
FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# system deps (slim imaj için temel araçlar)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl ca-certificates \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# requirements önce kopyalanır ki layer cache kullanılsın
COPY requirements.txt ./requirements.txt
RUN pip install --upgrade pip && pip install -r requirements.txt

# app dosyaları
COPY . .

# oluşturulacak dizinler (persist edeceğiz)
RUN mkdir -p uploads models data

# gunicorn prod server (flask dev server yerine)
RUN pip install gunicorn==21.2.0

EXPOSE 8000

# OLLAMA_URL opsiyonel (LLM enrichment kapalı ise gerekmez)
# ENV OLLAMA_URL=http://host.docker.internal:11434

# healthcheck (opsiyonel)
HEALTHCHECK --interval=30s --timeout=5s --start-period=20s --retries=3 \
  CMD python -c "import socket; s=socket.socket(); s.connect(('127.0.0.1',8000))" || exit 1

# production command
CMD ["gunicorn", "-w", "2", "-b", "0.0.0.0:8000", "app:app"]
