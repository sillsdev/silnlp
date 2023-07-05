ARG CUDA_VERSION=11.2.2-cudnn8-runtime-ubuntu20.04
ARG PYTHON_VERSION=3.8
ARG POETRY_VERSION=1.5.1

FROM python:$PYTHON_VERSION-slim as builder
ARG POETRY_VERSION

ENV POETRY_HOME=/opt/poetry
ENV POETRY_VENV=/opt/poetry-venv
ENV POETRY_CACHE_DIR=/opt/.cache

# Install poetry separated from system interpreter
RUN python3 -m venv $POETRY_VENV \
    && $POETRY_VENV/bin/pip install -U pip setuptools \
    && $POETRY_VENV/bin/pip install poetry==${POETRY_VERSION}

# Add `poetry` to PATH
ENV PATH="${PATH}:${POETRY_VENV}/bin"

WORKDIR /src
COPY . /src
RUN poetry build
RUN poetry export --without-hashes -f requirements.txt > requirements.txt


FROM nvidia/cuda:$CUDA_VERSION
ARG PYTHON_VERSION

ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /root

# Install apt packages
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install -y \
    git \
    python$PYTHON_VERSION \
    python3-pip \
    wget \
    build-essential \
    gdb \
    curl \
    unzip

# Make some useful symlinks that are expected to exist
RUN ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python3  & \
    ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Install .NET SDK
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb
RUN apt-get update && \
    apt-get install -y dotnet-sdk-7.0
ENV DOTNET_ROLL_FORWARD=LatestMajor

# Install AWS CLI
RUN curl "https://awscli.amazonaws.com/awscli-exe-linux-x86_64.zip" -o "awscliv2.zip" && \
    unzip awscliv2.zip && \
    ./aws/install && \
    rm awscliv2.zip

RUN rm -rf /var/lib/apt/lists/*

COPY --from=builder /src/requirements.txt .
RUN --mount=type=cache,target=/root/.cache pip install -r requirements.txt && rm requirements.txt

ENTRYPOINT [ "/bin/bash", "-it" ]