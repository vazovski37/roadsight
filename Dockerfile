# CPU-only demo image: FastAPI + ONNX Runtime, no PyTorch.
#   docker build -t roadsight .
#   docker run --rm -p 8000:8000 roadsight        -> http://localhost:8000
FROM python:3.11-slim
WORKDIR /srv
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 && rm -rf /var/lib/apt/lists/*
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt
# only the torch-free modules are needed at serving time
COPY roadsight/__init__.py roadsight/classes.py roadsight/labels.py roadsight/postprocess.py roadsight/
COPY app/ app/
COPY models/roadsight_int8.onnx models/roadsight_int8.json models/
ENV ROADSIGHT_MODEL=models/roadsight_int8.onnx ROADSIGHT_THREADS=4
EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
