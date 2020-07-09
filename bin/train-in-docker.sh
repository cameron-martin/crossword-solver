#!/usr/bin/env bash

set -eo pipefail

mkdir -p tmp

IMAGE_ID_FILE=tmp/docker-image-id
CONTAINER_ID_FILE=tmp/docker-container-id

USER_ID=$(id -u $SUDO_USER)
GROUP_ID=$(id -g $SUDO_USER)

if [ ! -f "$IMAGE_ID_FILE" ]; then
    docker build --iidfile "$IMAGE_ID_FILE" --build-arg user_id=$USER_ID --build-arg group_id=$GROUP_ID .
fi

IMAGE_ID="$(cat "$IMAGE_ID_FILE")"

if [ ! -f "$CONTAINER_ID_FILE" ]; then
    docker create -it -u $USER_ID:$GROUP_ID -v $(realpath .):/project --workdir /project --gpus all --cidfile "$CONTAINER_ID_FILE" "$IMAGE_ID" ./bin/run-container.sh
fi

CONTAINER_ID="$(cat "$CONTAINER_ID_FILE")"

docker start -a "$CONTAINER_ID"