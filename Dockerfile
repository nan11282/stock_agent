FROM python:3.12-slim

WORKDIR /app

RUN pip install --no-cache-dir anthropic openai chromadb pysqlite3-binary jieba akshare schedule

COPY . .

CMD ["python", "main.py"]
