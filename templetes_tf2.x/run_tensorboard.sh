#!/bin/sh
IMAGE_NAME=tensorflow2-image
CONTAINER_NAME=tensorboard-container
TENSORBOARD_DIR=tensorboard
PORT=6006
USE_DEBUGGER=0
PORT_DEBUGGER=6007

if [ ! "$(docker image ls -q ${IMAGE_NAME})" ]; then
    docker build ../docker -t ${IMAGE_NAME} -f ../docker/dockerfile_tf14
fi

mkdir -p ${TENSORBOARD_DIR}
docker rm -f ${CONTAINER_NAME}
docker run -d -it --rm -p ${PORT}:${PORT} -v ${HOME}/machine-learning_exercises_tensorflow:/mnt/machine-learning_exercises_tensorflow --name ${CONTAINER_NAME} --runtime=nvidia ${IMAGE_NAME} /bin/bash

if [ ${USE_DEBUGGER} = 0 ]; then
    docker exec -it ${CONTAINER_NAME} /bin/sh -c "cd /mnt/machine-learning_exercises_tensorflow/templetes_tf1.4 && \
        tensorboard --logdir ${TENSORBOARD_DIR} --port ${PORT} --bind_all --debugger_port ${PORT}"
else
    docker exec -it ${CONTAINER_NAME} /bin/sh -c "cd /mnt/machine-learning_exercises_tensorflow/templetes_tf1.4 && \
        tensorboard --logdir ${TENSORBOARD_DIR} --port ${PORT} --bind_all --debugger_port ${PORT_DEBUGGER}"
fi
