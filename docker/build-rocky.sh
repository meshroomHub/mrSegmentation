#!/bin/bash
set -ex

test -z "$MRSEGMENTATION_VERSION" && MRSEGMENTATION_VERSION="$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD)"
test -z "$CUDA_VERSION" && CUDA_VERSION=12.1.1
test -z "$ROCKY_VERSION" && ROCKY_VERSION=9

test -d docker || (
    echo This script must be run from the top level directory.
    exit 1
)

# Meshroom
docker build \
    --rm \
    --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
    --build-arg "ROCKY_VERSION=${ROCKY_VERSION}" \
    --tag "alicevision/mrsegmentation:${MRSEGMENTATION_VERSION}-rocky${ROCKY_VERSION}-cuda${CUDA_VERSION}" \
    -f docker/Dockerfile_rocky .
