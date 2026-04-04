FROM python:3.10-slim

# Use a non-root user for security and HF compatibility
RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH

# Set working directory
WORKDIR $HOME/app

# System deps (curl for healthcheck)
# Needs to be root for apt-get
USER root
RUN apt-get update && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/*
USER user

# Install Python dependencies first (layer caching)
COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Copy full source
COPY --chown=user . .

# Ensure package __init__ files exist
RUN touch src/__init__.py api/__init__.py

# Create data directory for SQLite database
RUN mkdir -p $HOME/app/data && chown -R user:user $HOME/app/data

# Environment defaults
ENV SUPPORT_ENV_DB=$HOME/app/data/support_env.db
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
