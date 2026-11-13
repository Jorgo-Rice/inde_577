FROM ghcr.io/astral-sh/uv:python3.12-bookworm-slim

WORKDIR /app
COPY pyproject.toml uv.lock README.md LICENSE ./
COPY src ./src
COPY configs ./configs
RUN uv sync --locked --no-dev
RUN uv run dsm-project train-example --output artifacts/example-classifier

EXPOSE 8000
CMD ["uv", "run", "dsm-project", "serve", "--artifact", "artifacts/example-classifier", "--host", "0.0.0.0", "--port", "8000"]
