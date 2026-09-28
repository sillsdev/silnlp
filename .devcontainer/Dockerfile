ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.17
FROM ubuntu:24.04
ARG PYTHON_VERSION
WORKDIR /app
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
RUN curl -LsSf https://astral.sh/uv/install.sh | sh
# Make some useful symlinks that are expected to exist
RUN ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python3  & \
    ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python
# Clean up
RUN rm -rf /var/lib/apt/lists/*
# Set environment variables
ENV CLEARML_API_HOST="https://api.sil.hosted.allegro.ai"
ENV EFLOMAL_PATH=/workspaces/silnlp/.venv/lib/python3.12/site-packages/eflomal/bin
CMD ["bash"]
