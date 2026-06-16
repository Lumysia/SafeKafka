FROM python:3.10-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    PATH="/app/.venv/bin:$PATH"

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml uv.lock* requirements.txt ./
RUN pip install --upgrade pip uv \
    && uv sync --no-dev --frozen --no-install-project

COPY safestream ./safestream
COPY scripts ./scripts
COPY previous_weights ./previous_weights

CMD ["uv", "run", "python", "-m", "safestream.dashboard"]
