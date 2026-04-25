FROM public.ecr.aws/docker/library/python:3.11-slim

# HuggingFace Spaces convention: container runs as uid 1000 with /home/user as HOME.
# Required for write access to mounted Spaces storage and to follow the platform
# guidance for non-root execution.
RUN useradd --create-home --uid 1000 user
WORKDIR /home/user/app

# Install Python deps as root (writes into system site-packages), then drop privileges.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY --chown=user:user . .

USER user

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PORT=7860 \
    HOME=/home/user

EXPOSE 7860

# HF Spaces health probe hits / and /health; the explicit HEALTHCHECK gives
# `docker run` users the same signal locally.
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; \
    sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:'+__import__('os').environ.get('PORT','7860')+'/health',timeout=4).status==200 else 1)"

# Exec-form CMD via `sh -c` so $PORT still expands at runtime (HF Spaces always
# sets PORT=7860, but other platforms may inject a different value) while keeping
# proper signal forwarding to PID 1.
CMD ["sh", "-c", "uvicorn server:app --host 0.0.0.0 --port ${PORT}"]
