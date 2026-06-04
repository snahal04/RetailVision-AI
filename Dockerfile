FROM python:3.11-slim

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements-api.txt requirements.txt ./
RUN pip install --no-cache-dir -r requirements-api.txt

COPY app ./app
COPY pipeline ./pipeline
COPY store_layout.json ./

ENV PYTHONPATH=/app
ENV DATABASE_URL=sqlite:////app/data/store_intelligence.db

RUN mkdir -p /app/data /app/output

EXPOSE 8000

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
