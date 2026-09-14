FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# El build-arg queda fijado como ENV, así el default en runtime SIEMPRE coincide
# con el modelo que se horneó en build-time: nunca hay descarga en runtime.
# Si necesitás otro modelo, reconstruí la imagen con --build-arg FACE_MODEL_NAME=...
# en vez de cambiar la env var en un `docker run` de esta misma imagen.
ARG FACE_MODEL_NAME=buffalo_l
ENV FACE_MODEL_NAME=${FACE_MODEL_NAME}
ENV FACE_MODELS_ROOT=/app/models

RUN python -c "\
from insightface.app import FaceAnalysis; \
FaceAnalysis(name='${FACE_MODEL_NAME}', root='${FACE_MODELS_ROOT}').prepare(ctx_id=-1)"

COPY app/ ./app/

EXPOSE 8000

CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000} --workers ${UVICORN_WORKERS:-1}"]
