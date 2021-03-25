#!bin/bash
set -eu
#GPU_IDS=0
GPU_IDS=0,1
IMAGE_NAME=tensorflow2.4-image
CONTAINER_NAME=tensorflow2.4-container
EXPER_NAME=debug_2gpus
TENSORBOARD_DIR=tensorboard

mkdir -p ${TENSORBOARD_DIR}
sudo rm -rf ${TENSORBOARD_DIR}/${EXPER_NAME}
sudo rm -rf ${TENSORBOARD_DIR}/${EXPER_NAME}_valid
sudo rm -rf ${TENSORBOARD_DIR}/${EXPER_NAME}_debug

if [ ! "$(docker image ls -q ${IMAGE_NAME})" ]; then
    docker build ../docker -t ${IMAGE_NAME} -f ../docker/dockerfile_tf2x
fi
if [ "$(docker ps -aqf "name=${CONTAINER_NAME}")" ]; then
    docker rm -f ${CONTAINER_NAME}
fi

docker run -d -it -v ${HOME}/machine-learning_exercises_tensorflow:/mnt/machine-learning_exercises_tensorflow --name ${CONTAINER_NAME} --runtime=nvidia ${IMAGE_NAME} /bin/bash
docker exec -it ${CONTAINER_NAME} /bin/sh -c "cd /mnt/machine-learning_exercises_tensorflow/templetes_tf2.x_with_keras && \
    python train_multi_gpu.py \
        --exper_name ${EXPER_NAME} \
        --n_diaplay_step 10 \
        --gpu_ids ${GPU_IDS} \
        --debug"
