#!/bin/sh
#conda activate tensorflow23_py36
set -eu

EXPER_NAME=debug
rm -rf tensorboard/${EXPER_NAME}
rm -rf tensorboard/${EXPER_NAME}_valid
if [ ${EXPER_NAME} = "debug" ] ; then
    N_DISPLAY_STEP=10
    N_DISPLAY_VALID_STEP=50
else
    N_DISPLAY_STEP=100
    N_DISPLAY_VALID_STEP=500
fi

python train.py \
    --exper_name ${EXPER_NAME} \
    --n_diaplay_step 10 --n_display_valid_step 10 --n_save_epoches 100 \
    --debug
