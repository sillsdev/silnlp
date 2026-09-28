ARG PYTHON_VERSION=3.12
ARG UV_VERSION=0.12.17

FROM ghcr.io/astral-sh/uv:${UV_VERSION} AS uv

FROM ubuntu:24.04

ARG PYTHON_VERSION=3.12

ENV PIP_DISABLE_PIP_VERSION_CHECK=on
ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /root

# Install apt packages
RUN apt-get update \
    && apt-get upgrade -y \
    && apt-get install --no-install-recommends -y \
    ca-certificates \
    git \
    python$PYTHON_VERSION \
    python$PYTHON_VERSION-venv \
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

COPY --from=uv /uv /uvx /bin/

# Make some useful symlinks that are expected to exist
RUN ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python3  & \
    ln -sfn /usr/bin/python${PYTHON_VERSION} /usr/bin/python

# Install locked runtime dependencies with uv
WORKDIR /tmp/silnlp-deps
ENV UV_PROJECT_ENVIRONMENT=/opt/venv
ENV PATH="/opt/venv/bin:$PATH"
COPY pyproject.toml uv.lock ./
RUN uv sync --locked --no-dev --no-install-project \
    && rm pyproject.toml uv.lock
WORKDIR /root

# Set eflomal path
ENV EFLOMAL_PATH=/opt/venv/lib/python3.12/site-packages/eflomal/bin

# Install fast_align
RUN apt-get update && \
    apt-get install --no-install-recommends -y libgoogle-perftools-dev libsparsehash-dev
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

# Clone silnlp and make it the starting directory
RUN git clone https://github.com/sillsdev/silnlp.git
WORKDIR /root/silnlp

# Default docker run behavior
CMD [ "/bin/bash", "-it" ]