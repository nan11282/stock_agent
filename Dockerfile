FROM python:3.12-slim

WORKDIR /app

ENV HF_HOME=/root/.cache/huggingface \
    RERANK_ENABLED=1 \
    RERANK_MODEL=BAAI/bge-reranker-base \
    RERANK_POOL_SIZE=12

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "app/main.py"]
