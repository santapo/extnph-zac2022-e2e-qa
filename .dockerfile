FROM nvidia/cuda:11.6.2-base-ubuntu20.04

RUN  apt-get update \
    && apt-get install -y wget curl python3.8 python3-pip rsync \
    && rm -rf /var/lib/apt/lists/*

# USER extNPH
COPY requirements.txt /tmp/
RUN --mount=type=cache,target=/root/.cache/pip pip install --no-cache-dir -r /tmp/requirements.txt

COPY data /code/data

RUN --mount=target=/ctx rsync -r --exclude="data" \
                                /ctx/ /code
WORKDIR /code
CMD ./scripts/prepare_elasticsearch.sh