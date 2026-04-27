FROM python:3.11-slim

WORKDIR /app

# Cài đặt uv
RUN pip install uv

# Copy cấu hình project và cài đặt siêu tốc bằng uv
COPY pyproject.toml README.md* ./
RUN uv pip install --system .

# Copy phần mã nguồn còn lại vào container
COPY . .

# Đặt PYTHONPATH để Python nhận diện thư mục src
ENV PYTHONPATH=/app