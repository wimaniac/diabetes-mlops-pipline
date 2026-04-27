FROM python:3.11-slim

WORKDIR /app

RUN pip install uv

COPY pyproject.toml README.md* ./
RUN uv pip install --system .

COPY . .

ENV PYTHONPATH=/app
