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
RUN pip install uv && uv pip install --system --no-deps .

# Copy the rest of the code
COPY . .

# Install supervisord
RUN pip install supervisor

# Copy supervisord config
COPY supervisord.conf /etc/supervisord.conf

# Expose ports: 8501 for Streamlit, 5000 for FastAPI
EXPOSE 8501 5000

CMD ["/usr/local/bin/supervisord", "-c", "/etc/supervisord.conf"]
