FROM python:3.10

WORKDIR /app
COPY . .

RUN pip install --no-cache-dir -r requirements.txt

CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port $PORT"]