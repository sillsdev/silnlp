ARG CUDA_VERSION=11.2.2-cudnn8-runtime-ubuntu20.04
ARG PYTHON_VERSION=3.8
ARG POETRY_VERSION=1.5.1
FROM nvidia/cuda:$CUDA_VERSION
ARG PYTHON_VERSION
ARG POETRY_VERSION
WORKDIR /app
ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache
ENV DOTNET_ROLL_FORWARD=LatestMajor
ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone
# Install .NET SDK
RUN apt-get update
RUN apt-get install --no-install-recommends -y wget
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb
# Install apt packages
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install --no-install-recommends -y \
    git \
    python$PYTHON_VERSION \
    python3-pip \
    python$PYTHON_VERSION-venv \
    build-essential \
    gdb \
    curl \
    unzip \
    dotnet-sdk-7.0
# Make some useful symlinks that are expected to exist
RUN ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python3  & \
    ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python
# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}
# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"
# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm awscliv2.zip
RUN rm -rf /var/lib/apt/lists/*
ENTRYPOINT [ "/bin/bash", "-it" ]