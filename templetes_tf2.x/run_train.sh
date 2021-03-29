#!bin/bash
set -eu
IMAGE_NAME=tensorflow2.4-image
CONTAINER_NAME=tensorflow2.4-container
EXPER_NAME=debug
LOAD_CHECKPOINTS_DIR="checkpoints/${EXPER_NAME}/"
TENSORBOARD_DIR=tensorboard

mkdir -p ${TENSORBOARD_DIR}
sudo rm -rf ${TENSORBOARD_DIR}/${EXPER_NAME}
sudo rm -rf ${TENSORBOARD_DIR}/${EXPER_NAME}_valid
sudo rm -rf ${TENSORBOARD_DIR}/${EXPER_NAME}_debug
#sudo rm -rf checkpoints

if [ ! "$(docker image ls -q ${IMAGE_NAME})" ]; then
    docker build ../docker -t ${IMAGE_NAME} -f ../docker/dockerfile_tf2x
fi

if [ "$(docker ps -aqf "name=${CONTAINER_NAME}")" ]; then
    docker rm -f ${CONTAINER_NAME}
fi

docker run -d -it -v ${HOME}/machine-learning_exercises_tensorflow:/mnt/machine-learning_exercises_tensorflow --name ${CONTAINER_NAME} --runtime=nvidia ${IMAGE_NAME} /bin/bash
docker exec -it ${CONTAINER_NAME} /bin/sh -c "cd /mnt/machine-learning_exercises_tensorflow/templetes_tf2.x && \
    python train.py \
        --exper_name ${EXPER_NAME} \
        --n_diaplay_step 10 \
        --debug"

#        --load_checkpoints_dir ${LOAD_CHECKPOINTS_DIR} \
#        --use_tensorboard_debugger \
