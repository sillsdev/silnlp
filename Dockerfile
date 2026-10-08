ARG PYTHON_VERSION=3.12

FROM ubuntu:24.04

ARG PYTHON_VERSION=3.12

COPY --from=ghcr.io/astral-sh/uv:0.12.23 /uv /uvx /bin/

ENV TZ=America/New_York
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

WORKDIR /root

# Install apt packages
RUN apt-get update
RUN apt-get upgrade -y
RUN apt-get install --no-install-recommends -y \
    ca-certificates \
    git \
    python$PYTHON_VERSION \
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

# Install locked runtime dependencies
WORKDIR /tmp/silnlp
COPY pyproject.toml uv.lock ./
# Hashes keep the cross-index lookup safe for the CUDA-specific torch wheel.
RUN uv export --locked --no-dev --no-emit-project --emit-index-url --format requirements-txt --output-file requirements.txt \
    && uv pip install --system --break-system-packages --index-strategy unsafe-best-match --require-hashes \
        --requirements requirements.txt \
    && rm requirements.txt

# Install fast_align
WORKDIR /root
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