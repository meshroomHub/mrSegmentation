#!/bin/bash
set -ex

test -z "$MRSEGMENTATION_VERSION" && MRSEGMENTATION_VERSION="$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD)"
test -z "$CUDA_VERSION" && CUDA_VERSION=12.1.1
test -z "$UBUNTU_VERSION" && UBUNTU_VERSION=22.04

test -d docker || (
    echo This script must be run from the top level directory.
    exit 1
)

# Meshroom
docker build \
    --rm \
    --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
    --build-arg "UBUNTU_VERSION=${UBUNTU_VERSION}" \
    --tag "alicevision/mrsegmentation:${MRSEGMENTATION_VERSION}-ubuntu${UBUNTU_VERSION}-cuda${CUDA_VERSION}" \
    -f docker/Dockerfile_ubuntu .
