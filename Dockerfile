# Multi-stage Docker build for AI Trading Bot Production
FROM node:18-alpine AS frontend-builder

# Set working directory for frontend
WORKDIR /app/frontend

# Copy package files
COPY frontend/package*.json ./

# Install dependencies
RUN npm ci --only=production

# Copy frontend source
COPY frontend/ ./

# Build frontend for production
RUN npm run build

# Python backend stage
FROM python:3.11-slim AS backend-base

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1
ENV NODE_ENV=production
ENV TRADING_ENV=production

# Install system dependencies
RUN apt-get update && apt-get install -y \
    gcc \
    g++ \
    curl \
    postgresql-client \
    redis-tools \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for security
RUN useradd --create-home --shell /bin/bash trading && \
    mkdir -p /app && \
    chown -R trading:trading /app

# Set working directory
WORKDIR /app

# Copy requirements first for better caching
COPY backend/requirements.txt ./

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Switch to non-root user
USER trading

# Copy backend source code
COPY --chown=trading:trading backend/ ./backend/

# Copy built frontend
COPY --from=frontend-builder --chown=trading:trading /app/frontend/dist ./frontend/dist/

# Create necessary directories
RUN mkdir -p ./logs ./data ./models ./backups

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Expose ports
EXPOSE 8000 8080 9090

# Set default command
CMD ["python", "-m", "backend.trading_system_orchestrator"]

# Multi-service production image
FROM backend-base AS production

# Copy production configuration
COPY --chown=trading:trading docker/production/ ./config/

# Copy startup scripts
COPY --chown=trading:trading docker/scripts/ ./scripts/
RUN chmod +x ./scripts/*.sh

# Final production entrypoint
ENTRYPOINT ["./scripts/entrypoint.sh"]