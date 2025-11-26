# Dockerfile for Smart Forensic Investigation Platform
# Multi-stage build for optimized production image

# ============================================
# Stage 1: Backend Dependencies
# ============================================
FROM python:3.10-slim as backend-builder

LABEL maintainer="Forensic Platform Team"
LABEL description="Smart Forensic Investigation Platform - Backend Builder"

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    git \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements file
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# ============================================
# Stage 2: Frontend Build
# ============================================
FROM node:18-alpine as frontend-builder

LABEL description="Smart Forensic Investigation Platform - Frontend Builder"

WORKDIR /frontend

# Copy package files
COPY src/frontend/package*.json ./

# Install dependencies
RUN npm ci --silent

# Copy frontend source code
COPY src/frontend/ ./

# Build the frontend
RUN npm run build

# ============================================
# Stage 3: Final Production Image
# ============================================
FROM python:3.10-slim

LABEL maintainer="Forensic Platform Team"
LABEL version="1.0"
LABEL description="Smart Forensic Investigation Platform - Production Image"

# Set working directory
WORKDIR /app

# Install runtime dependencies only
RUN apt-get update && apt-get install -y \
    curl \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Create non-root user for security
RUN useradd -m -u 1000 forensic && \
    chown -R forensic:forensic /app

# Copy Python packages from builder
COPY --from=backend-builder /usr/local/lib/python3.10/site-packages /usr/local/lib/python3.10/site-packages
COPY --from=backend-builder /usr/local/bin /usr/local/bin

# Copy application source code
COPY --chown=forensic:forensic src/ ./src/

# Copy built frontend from frontend builder
COPY --from=frontend-builder --chown=forensic:forensic /frontend/dist ./src/frontend/dist

# Create necessary directories with proper permissions
RUN mkdir -p data/vector_db data/uploads data/cases && \
    chown -R forensic:forensic data/

# Copy data files if they exist
COPY --chown=forensic:forensic data* ./data/ 2>/dev/null || true

# Switch to non-root user
USER forensic

# Expose ports
EXPOSE 8000 3000

# Set environment variables
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    DEBUG_FORENSIC_BOT=false \
    PORT=8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=40s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Set the entry point
ENTRYPOINT ["python"]

# Default command to run the application
CMD ["src/api/case_api.py"]
