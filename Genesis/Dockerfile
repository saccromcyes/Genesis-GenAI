FROM python:3.11-slim

WORKDIR /app
COPY . /app

RUN pip install --no-cache-dir -r requirements.txt

# Note: model files are not included. Mount ./models at runtime.
CMD ["python", "-m", "genesis.cli", "--help"]
