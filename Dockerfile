ARG CUDA_VERSION="13.0.1"

FROM nvidia/cuda:${CUDA_VERSION}-cudnn-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    UV_LINK_MODE=copy \
    UV_PROJECT_ENVIRONMENT=/opt/venv \
    VIRTUAL_ENV=/opt/venv \
    PATH=/opt/venv/bin:/root/.local/bin:$PATH

RUN apt-get update && apt-get install -y --no-install-recommends \
        python3.12 python3.12-dev python3.12-venv \
        build-essential git curl ca-certificates \
        ninja-build cmake pkg-config \
        vim less openssh-client \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

WORKDIR /workspace

ENV UV_HTTP_TIMEOUT=600 \
    UV_CONCURRENT_DOWNLOADS=4

COPY pyproject.toml uv.lock* ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv venv --python python3.12 $VIRTUAL_ENV \
    && uv sync --frozen --no-install-project

CMD ["bash"]