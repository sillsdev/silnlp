ARG CUDA_VERSION=11.2.2-cudnn8-runtime-ubuntu20.04
ARG PYTHON_VERSION=3.8
ARG POETRY_VERSION=1.7.1

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
COPY poetry.lock pyproject.toml /src/
RUN poetry export -E eflomal --without-hashes -f requirements.txt > requirements.txt
COPY . /src
RUN poetry build

FROM nvidia/cuda:$CUDA_VERSION

ARG PYTHON_VERSION=3.8

ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /root

# Install apt packages
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install --no-install-recommends -y \
    git \
    python$PYTHON_VERSION \
    python3-pip \
    python3-dev \
    wget \
    build-essential \
    gdb \
    curl \
    unzip \
    nano \
    cmake \
    tar \
    && rm -rf /var/lib/apt/lists/* \
    && apt-get clean

# Make some useful symlinks that are expected to exist
RUN ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python3  & \
    ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Install .NET SDK
RUN wget https://packages.microsoft.com/config/ubuntu/20.04/packages-microsoft-prod.deb -O packages-microsoft-prod.deb && \
    dpkg -i packages-microsoft-prod.deb && \
    rm packages-microsoft-prod.deb
RUN apt-get update && \
    apt-get install --no-install-recommends -y dotnet-sdk-7.0
ENV DOTNET_ROLL_FORWARD=LatestMajor

# Install dependencies from poetry
COPY --from=builder /src/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && rm requirements.txt

# Set eflomal path
ENV EFLOMAL_PATH=/usr/local/lib/python3.8/dist-packages/eflomal/bin

# Install fast_align
RUN apt-get install --no-install-recommends -y libgoogle-perftools-dev libsparsehash-dev
RUN git clone https://github.com/clab/fast_align.git
RUN mkdir fast_align/build
RUN cmake -S fast_align -B fast_align/build
RUN make -C fast_align/build
RUN mv fast_align/build/atools fast_align/build/fast_align /usr/local/bin
RUN rm -rf fast_align
ENV FAST_ALIGN_PATH=/usr/local/bin

# Install mgiza
RUN apt-get install --no-install-recommends -y libboost-all-dev
RUN git clone https://github.com/moses-smt/mgiza.git
RUN cmake -S mgiza/mgizapp -B mgiza/mgizapp
RUN make -C mgiza/mgizapp
RUN make -C mgiza/mgizapp install
RUN mv mgiza/mgizapp/inst/mgiza mgiza/mgizapp/inst/mkcls mgiza/mgizapp/inst/plain2snt mgiza/mgizapp/inst/snt2cooc /usr/local/bin
RUN rm -rf mgiza
ENV MGIZA_PATH=/usr/local/bin

# Install meteor
RUN wget "https://download.oracle.com/java/21/latest/jdk-21_linux-x64_bin.tar.gz"
RUN tar -xf jdk-21_linux-x64_bin.tar.gz
RUN rm jdk-21_linux-x64_bin.tar.gz
RUN wget "http://www.cs.cmu.edu/~alavie/METEOR/download/meteor-1.5.tar.gz"
RUN tar -xf meteor-1.5.tar.gz
RUN rm meteor-1.5.tar.gz
RUN mv meteor-1.5/meteor-1.5.jar /usr/local/bin
RUN rm -rf meteor-1.5
ENV METEOR_PATH=/usr/local/bin

# Other environment variables
ENV SIL_NLP_DATA_PATH=/silnlp
RUN mkdir -p .cache/silnlp
ENV SIL_NLP_CACHE_EXPERIMENT_DIR=/root/.cache/silnlp
ENV CLEARML_API_HOST="https://api.sil.hosted.allegro.ai"

# Clone silnlp and make it the starting directory
RUN git clone https://github.com/sillsdev/silnlp.git
WORKDIR /root/silnlp

# Default docker run behavior
CMD [ "/bin/bash", "-it" ]