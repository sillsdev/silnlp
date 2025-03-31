ARG PYTHON_VERSION=3.10
ARG POETRY_VERSION=1.7.1
FROM ubuntu:22.04
ARG PYTHON_VERSION
ARG POETRY_VERSION
WORKDIR /app
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Install apt packages
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install --no-install-recommends -y \
    git \
    python$PYTHON_VERSION \
    python3-pip \
    python3-dev \
    python$PYTHON_VERSION-venv \
    build-essential \
    gdb \
    curl \
    unzip
RUN apt-get update && apt-get install --no-install-recommends -y \
    fuse3 \
    rclone
# Make some useful symlinks that are expected to exist
RUN ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python3  & \
    ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python
# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}
# Add `poetry` to PATH and configure
ENV PATH="${PATH}:${POETRY_VENV}/bin"
RUN poetry config virtualenvs.create true && \
    poetry config virtualenvs.in-project true
# Clean up
RUN rm -rf /var/lib/apt/lists/*
# Create caches
RUN mkdir -p /root/.cache/silnlp/experiments
RUN mkdir /root/.cache/silnlp/projects
ENV SIL_NLP_CACHE_EXPERIMENT_DIR=/root/.cache/silnlp/experiments
ENV SIL_NLP_CACHE_PROJECT_DIR=/root/.cache/silnlp/projects
# Set up the MinIO/Backblaze bucket
RUN mkdir -p ~/M
RUN mkdir -p ~/B
RUN mkdir -p ~/.config/rclone
# Set environment variables
ENV CLEARML_API_HOST="https://api.sil.hosted.allegro.ai"
ENV SIL_NLP_DATA_PATH=/silnlp
ENV EFLOMAL_PATH=/workspaces/silnlp/.venv/lib/python3.10/site-packages/eflomal/bin
CMD ["bash"]
