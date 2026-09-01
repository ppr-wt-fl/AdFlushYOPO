# syntax=docker/dockerfile:1
#
# Dockerfile for source/autokeras/train_autokeras.py -- an AutoKeras
# StructuredDataClassifier search for AdFlush, exported to a tf.js
# LayersModel that can keep training in the browser. AutoML alternative to
# source/train_optuna.py's Optuna pipeline.
#
# GPU base image: bundles a CUDA/cuDNN runtime matched to TensorFlow 2.15,
# so this assumes an NVIDIA driver + the NVIDIA Container Toolkit on the
# host (`--gpus all` at `docker run` time). Runs CPU-only too, but the full
# search below is far too slow for that in practice.
#
# ---------------------------------------------------------------------------
# Build layout -- multi-stage, ordered so a code edit rebuilds ~nothing
# ---------------------------------------------------------------------------
#   node      pinned Node.js toolchain      -> keyed on NODE_VERSION only
#   npm-deps  tfjs-node + native binary     -> keyed on package.json only
#   py-deps   AutoKeras / TF / tfjs stack   -> keyed on requirements_autokeras.txt only
#   runtime   FROM py-deps; COPY node + node_modules in; then dataset, then code
#
# Under BuildKit the node/npm-deps and py-deps chains build in parallel.
# On a rebuild:
#   * edit train_autokeras.py  -> only the final ~1 KB COPY re-runs
#   * edit tfjs_validate/validate.js -> only its COPY re-runs (npm install cached)
#   * edit requirements        -> only py-deps (+ runtime assembly) re-runs
#   * bump NODE_VERSION        -> only node/npm-deps re-run
# The repo-root .dockerignore is large; source/autokeras/Dockerfile.dockerignore
# overrides it for this build (BuildKit) to an allowlist of just the files
# below, so the build context is a few hundred MB, not ~2 GB.
#
# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
# Everything external is pinned to an exact, content-addressed version:
#   * base image      -> pinned by sha256 digest (not just the :2.15.0-gpu tag)
#   * Python packages -> every direct dep pinned to == in requirements_autokeras.txt
#   * pip/setuptools/wheel -> pinned versions
#   * Node.js         -> exact release tarball from nodejs.org, sha256-verified
#                        (NOT the NodeSource apt repo, whose versions rotate)
#   * npm packages    -> @tensorflow/tfjs-node pinned to == in package.json
#
# Still floating (lock further if you need bit-identical rebuilds):
#   * apt packages (curl/ca-certificates/xz-utils) from the Ubuntu archive at
#     build time -- pin with `pkg=version` + snapshot.ubuntu.com if required
#   * transitive Python deps -- generate a hash lock from the first good build
#     (`pip freeze > requirements_autokeras.lock` in the py-deps stage) and
#     install that instead
#   * transitive npm deps -- run `npm install` once, commit the generated
#     source/tfjs_validate/package-lock.json, then switch the `npm install`
#     in the npm-deps stage to `npm ci`
#
# Bit-exact *training* results on GPU are separate: the script seeds
# everything (tf.keras.utils.set_random_seed), but full determinism also
# needs TF_DETERMINISTIC_OPS=1 / TF_CUDNN_DETERMINISTIC=1 (set them at
# `docker run` time). Left unset here because a missing deterministic op
# kernel raises, which would be a bad way to lose a multi-day run.
#
# ---------------------------------------------------------------------------
# Build (from the REPO ROOT; needs BuildKit, the default in modern Docker):
#   docker build -f source/autokeras/Dockerfile -t adflush-autokeras .
#
# Run (full dataset baked in; model/ + output/ bind-mounted so results land
# on the host). Deep search -- budget ~9-10 days:
#   docker run --rm --gpus all \
#     -v "$(pwd)/source/autokeras/model:/adflush/model" \
#     -v "$(pwd)/source/autokeras/output:/adflush/output" \
#     adflush-autokeras
# ---------------------------------------------------------------------------

ARG BASE_IMAGE=tensorflow/tensorflow:2.15.0-gpu@sha256:66b44c162783bb92ab6f44c1b38bcdfef70af20937089deb7bc20a4f3d7e5491

# ===========================================================================
# Stage: node -- pinned Node.js toolchain, sha256-verified. Independent of
# every Python dep and every line of app code; rebuilds only on NODE_VERSION.
# ===========================================================================
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

# ===========================================================================
# Stage: npm-deps -- @tensorflow/tfjs-node and its prebuilt native binary.
# Only package.json is copied, so editing validate.js does NOT bust this.
# ===========================================================================
FROM node AS npm-deps

WORKDIR /build/tfjs_validate
COPY source/tfjs_validate/package.json ./
RUN npm install --no-audit --no-fund --loglevel=error

# ===========================================================================
# Stage: py-deps -- the AutoKeras / TensorFlow 2.15 / tfjs Python stack.
# Rebuilds only when requirements_autokeras.txt changes. Builds in parallel
# with node/npm-deps under BuildKit.
# ===========================================================================
FROM ${BASE_IMAGE} AS py-deps

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

COPY source/autokeras/requirements_autokeras.txt /tmp/requirements_autokeras.txt
RUN python3 -m pip install --no-cache-dir \
        "pip==24.3.1" "setuptools==75.6.0" "wheel==0.45.1" && \
    python3 -m pip install --no-cache-dir -r /tmp/requirements_autokeras.txt

# ===========================================================================
# Stage: runtime -- inherit the Python deps layer, pull in the Node runtime
# and node_modules by copy, then add data and code (small, last, cache-cheap).
# ===========================================================================
FROM py-deps AS runtime

ENV PYTHONUNBUFFERED=1 \
    PYTHONHASHSEED=0 \
    TF_CPP_MIN_LOG_LEVEL=2 \
    LANG=C.UTF-8 \
    TZ=UTC

# validate.js is run as `node validate.js`; npm is not needed at run time,
# so only the self-contained node binary is copied.
COPY --from=node /opt/node/bin/node /usr/local/bin/node

WORKDIR /adflush

# node_modules (rarely changes) before the script that imports it.
COPY --from=npm-deps /build/tfjs_validate/node_modules source/tfjs_validate/node_modules
COPY source/tfjs_validate/package.json source/tfjs_validate/validate.js source/tfjs_validate/

# Dataset (large, stable) before app code (small, edited often) so a code
# edit never re-copies the CSVs.
COPY dataset/AdFlush_train.csv dataset/AdFlush_test.csv dataset/
COPY source/autokeras/train_autokeras.py source/train_autokeras.py

RUN mkdir -p model output

WORKDIR /adflush/source
ENTRYPOINT ["python3", "train_autokeras.py"]
