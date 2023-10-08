FROM nvidia/cuda:12.2.0-devel-ubuntu22.04
ENV CUDA_HOME=/usr/local/cuda
WORKDIR /app

# Install required packages
RUN set -xe \
 && apt-get -y update \
 && apt-get install -y software-properties-common curl build-essential git libaio-dev llvm-11 clang wget \
 && apt-get -y update \
 && add-apt-repository universe \
 && apt-get -y update \
 && apt-get -y install python3 python3-pip \
 && apt-get clean

# Install torch
RUN set -xe \
 && pip install --upgrade pip \
 && pip install torch==2.0.1

# Install other dependencies
RUN set -xe \
 && pip install -r requirements.txt

# Copy source code
COPY . .

ENTRYPOINT ["sleep", "inf"]
