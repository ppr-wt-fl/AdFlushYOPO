# Dockerfile for source/autokeras/train_autokeras.py -- an
# AutoKeras (keras-team/autokeras) StructuredDataClassifier search for
# AdFlush, exported to a tf.js LayersModel that can keep training in the
# browser. Feasibility check for AutoKeras as an AutoML alternative to
# source/train_optuna.py's Optuna-based pipeline.
#
# GPU base image: bundles a CUDA/cuDNN runtime matched to TensorFlow 2.15,
# so this assumes an NVIDIA driver + the NVIDIA Container Toolkit on the
# host (`--gpus all` at `docker run` time) for a real run. TensorFlow uses
# whichever GPU is visible automatically -- no special flags needed. This
# was built and smoke-tested WITHOUT --gpus all (CPU only, TF falls back
# automatically) on a dev machine with no GPU.
#
# Build (from the REPO ROOT, so dataset/ and source/tfjs_validate can be
# copied in):
#   docker build -f source/autokeras/Dockerfile \
#     -t adflush-autokeras .
#
# Run (uses the bundled dataset/*_sample.csv baked into the image; model/
# output are bind-mounted so results land on the host):
#   docker run --rm \
#     -v "$(pwd)/source/autokeras/model:/adflush/model" \
#     -v "$(pwd)/source/autokeras/output:/adflush/output" \
#     adflush-autokeras

FROM tensorflow/tensorflow:2.15.0-gpu

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    TF_CPP_MIN_LOG_LEVEL=2

# Node.js (LTS) for the tfjs-node round-trip validation step.
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends curl ca-certificates gnupg && \
    curl -fsSL https://deb.nodesource.com/setup_20.x | bash - && \
    apt-get install -y --no-install-recommends nodejs && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /adflush

COPY ./source/autokeras/requirements_autokeras.txt requirements_autokeras.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip && \
    python3 -m pip install --no-cache-dir -r requirements_autokeras.txt

COPY dataset/AdFlush_train_sample.csv dataset/AdFlush_train_sample.csv
COPY dataset/AdFlush_test_sample.csv dataset/AdFlush_test_sample.csv

COPY source/tfjs_validate source/tfjs_validate
RUN cd source/tfjs_validate && npm install

COPY source/autokeras/train_autokeras.py source/train_autokeras.py

RUN mkdir -p model output

WORKDIR /adflush/source
ENTRYPOINT ["python3", "train_autokeras.py"]
