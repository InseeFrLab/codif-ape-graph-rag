# Use an official lightweight Python image
FROM python:3.11-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Install uv (if needed)
COPY uv.lock pyproject.toml ./
# Install uv package manager
RUN pip install uv

# Sync dependencies
RUN uv sync

# Copy the rest of the code
COPY . .

RUN pip install supervisor

# Expose ports: 8501 for Streamlit, 5000 for FastAPI
EXPOSE 8501 5000

ENV PYTHONPATH=src/
# CMD ["uv", "run", "uvicorn", "src.api.main:app", "--host", "0.0.0.0", "--port", "5000"]
# CMD ["uv", "run", "streamlit", "run", "app/main.py"]
CMD ["supervisord", "-c", "supervisord.conf"]