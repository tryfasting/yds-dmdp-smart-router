# ==========================================
# Stage 1: Build & Dependency Resolution using uv
# ==========================================
FROM python:3.11-slim AS builder

WORKDIR /app

# Install uv package manager
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Copy configuration files
COPY pyproject.toml .

# Install dependencies into a virtual environment
RUN uv venv /opt/venv && \
    uv pip compile pyproject.toml -o requirements.txt && \
    uv pip install --no-cache -r requirements.txt --target /opt/venv/lib/python3.11/site-packages

# ==========================================
# Stage 2: Final Light Runtime Image
# ==========================================
FROM python:3.11-slim AS runner

WORKDIR /app

# Copy virtual environment from builder stage
COPY --from=builder /opt/venv/lib/python3.11/site-packages /usr/local/lib/python3.11/site-packages

# Copy source code and UI static files
COPY src/ /app/src/

# Copy model assets (in production, they can be baked in or mounted)
COPY models/ /app/models/

# Environmental setups
ENV PYTHONPATH="/app/src" \
    PORT=8000 \
    HOST="0.0.0.0"

# L7 Healthcheck validation
EXPOSE 8000

# Start FastAPI application using Uvicorn
CMD ["python", "-m", "uvicorn", "smartrouter.main:app", "--host", "0.0.0.0", "--port", "8000"]
