# ---------------------------------------------------------------------------
# Build from the REPO ROOT:
#   docker build -f source/autokeras/Dockerfile -t adflush-autokeras .
#
# Run (dataset/ + runs/ bind-mounted -> each run lands in runs/<timestamp>/ on host)
#   docker run --rm --gpus all \
#     -v "$(pwd)/dataset:/adflush/dataset" \
#     -v "$(pwd)/source/autokeras/runs:/adflush/runs" \
#     adflush-autokeras
# ---------------------------------------------------------------------------

ARG BASE_IMAGE=tensorflow/tensorflow:2.15.0-gpu@sha256:66b44c162783bb92ab6f44c1b38bcdfef70af20937089deb7bc20a4f3d7e5491

#pinned Node.js 
FROM ${BASE_IMAGE} AS node

ENV DEBIAN_FRONTEND=noninteractive \
    NODE_VERSION=20.18.1 \
    NODE_SHA256=c6fa75c841cbffac851678a472f2a5bd612fff8308ef39236190e1f8dbb0e567

RUN apt-get update && \
    apt-get install -y --no-install-recommends curl ca-certificates xz-utils && \
    rm -rf /var/lib/apt/lists/* && \
    curl -fsSLO "https://nodejs.org/dist/v${NODE_VERSION}/node-v${NODE_VERSION}-linux-x64.tar.xz" && \
    echo "${NODE_SHA256}  node-v${NODE_VERSION}-linux-x64.tar.xz" | sha256sum -c - && \
    mkdir -p /opt/node && \
    tar -xJf "node-v${NODE_VERSION}-linux-x64.tar.xz" -C /opt/node --strip-components=1 \
        --exclude CHANGELOG.md --exclude LICENSE --exclude README.md && \
    rm "node-v${NODE_VERSION}-linux-x64.tar.xz" && \
    /opt/node/bin/node --version
ENV PATH=/opt/node/bin:$PATH

# npm-deps 
FROM node AS npm-deps
WORKDIR /build/tfjs_validate
COPY source/tfjs_validate/package.json ./
RUN npm install --no-audit --no-fund --loglevel=error

# python deps
FROM ${BASE_IMAGE} AS py-deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY source/autokeras/requirements_autokeras.txt /tmp/requirements_autokeras.txt
RUN python3 -m pip install --no-cache-dir \
        "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1" && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements_autokeras.txt

# runtime
FROM py-deps AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    LANG=C.UTF-8 \
    TZ=UTC

# note, might be too overkill
# validate.js is run as `node validate.js` -> we just need node
COPY --from=node /opt/node/bin/node /usr/local/bin/node

WORKDIR /adflush

# node_modules (rarely changes) before the script that imports it.
COPY --from=npm-deps /build/tfjs_validate/node_modules source/tfjs_validate/node_modules
COPY source/tfjs_validate/package.json source/tfjs_validate/validate.js source/tfjs_validate/

COPY source/autokeras/train_autokeras.py source/train_autokeras.py

RUN mkdir -p dataset runs

WORKDIR /adflush/source
ENTRYPOINT ["python3", "train_autokeras.py"]
