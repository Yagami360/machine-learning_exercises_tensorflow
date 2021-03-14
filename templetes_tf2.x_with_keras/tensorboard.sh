#!/bin/sh
set -eu
TENSORBOARD_DIR=tensorboard
EXPER_NAME=debug
PORT=6006
USE_DEBUGGER=0

mkdir -p ${TENSORBOARD_DIR}
if [ ${USE_DEBUGGER} = 0 ]; then
    tensorboard --logdir ${TENSORBOARD_DIR} --port ${PORT} --bind_all
else
    tensorboard --logdir ${TENSORBOARD_DIR}/${EXPER_NAME}_debug --port ${PORT} --bind_all
fi
