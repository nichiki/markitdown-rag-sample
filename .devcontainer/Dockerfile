FROM mcr.microsoft.com/vscode/devcontainers/python:3.12

# venvを使わない
ENV UV_PROJECT_ENVIRONMENT=/home/vscode/.local

# uvをインストール
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uvx /bin/

# Chromium & 日本語フォントをインストール
RUN apt-get update && apt-get install -y \
    chromium \
    fonts-noto-cjk \
    fonts-noto-cjk-extra \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*