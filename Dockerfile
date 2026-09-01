FROM python:3.11-slim

WORKDIR /app

# Install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy project
COPY pyproject.toml .
COPY gunicorn_conf.py .
COPY src/ src/
COPY scripts/ scripts/

# Install package
RUN pip install -e .

# Data directory for FAISS index (mount as volume for persistence)
RUN mkdir -p /app/data

ENV VECTOR_STORE_PATH=/app/data/faiss_index
ENV PYTHONPATH=/app/src
ENV PYTHONUNBUFFERED=1

EXPOSE 8000

# Run API
CMD ["gunicorn", "-c", "gunicorn_conf.py", "knowledge_transfer_agent.api.main:app"]
