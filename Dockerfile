FROM python:3.10-slim

# Metadata
LABEL maintainer="Hackathon Team"
LABEL description="SupportEnv — AgentOps benchmark for LLM agent evaluation"
LABEL version="2.0.0"

# Set working directory
WORKDIR /app

# System deps (curl for healthcheck)
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (layer caching)
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy full source
COPY . .

# Ensure package __init__ files exist
RUN touch src/__init__.py api/__init__.py

# Create data directory for SQLite database
RUN mkdir -p /app/data

# Environment defaults
ENV SUPPORT_ENV_DB=/app/data/support_env.db
ENV ENABLE_SUPABASE=false
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Port 7860 — HuggingFace Spaces default
EXPOSE 7860

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:7860/health || exit 1

# Start FastAPI server
CMD ["uvicorn", "api.server:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
