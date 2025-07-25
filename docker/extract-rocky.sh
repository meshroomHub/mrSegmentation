#!/bin/bash
set -ex

test -z "$MRSEGMENTATION_VERSION" && MRSEGMENTATION_VERSION="$(git rev-parse --abbrev-ref HEAD)-$(git rev-parse --short HEAD)"
test -z "$CUDA_VERSION" && CUDA_VERSION="12.1.1"
test -z "$ROCKY_VERSION" && ROCKY_VERSION="9"

test -d docker || (
	echo This script must be run from the top level Meshroom directory
	exit 1
)

VERSION_NAME=${MRSEGMENTATION_VERSION}-rocky${ROCKY_VERSION}-cuda${CUDA_VERSION}

# Retrieve the Meshroom bundle folder
rm -rf ./plugins/mrSegmentation
CID=$(docker create alicevision/mrsegmentation:${VERSION_NAME})
docker cp ${CID}:/opt/MeshroomPlugin ./plugins/mrSegmentation
docker rm ${CID}
