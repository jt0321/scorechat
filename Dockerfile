# Runtime image for the demo deployment: the API plus the HTML/JS client.
#
# It carries no corpus. Everything an answer or a rendered score is built from
# lives in the database — the .krn submodule, data/mei and the commentary texts
# are ingestion inputs, and ingestion is not run here. That is why no stage
# fetches a submodule and why data/ is excluded in .dockerignore.
FROM python:3.12-slim

# gemini by default: a free tier that supports tool calling, which this pipeline
# requires. openrouter and cloudflare are OpenAI-compatible and need no extra.
ARG EXTRAS=gemini

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8080

WORKDIR /app

# Dependencies come from pyproject, so there is one list to keep correct. The
# package directories are copied first because hatchling resolves them at build.
COPY pyproject.toml ./
COPY analysis ./analysis
COPY commentary ./commentary
COPY db ./db
COPY pipeline ./pipeline
COPY frontend ./frontend
RUN pip install --no-cache-dir ".[${EXTRAS}]"

COPY server.py ./

# Nothing here writes to the filesystem, so the process needs no root.
RUN useradd --create-home --uid 10001 scorechat
USER scorechat

EXPOSE 8080
CMD ["python", "server.py"]
