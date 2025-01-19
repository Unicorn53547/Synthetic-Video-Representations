# Set the path to save checkpoints
OUTPUT_DIR=''
# Set the path to UCF101 train set. 
DATA_PATH='Annotations/YOUR_UCF_PATH'
MODEL_PATH='PATH_TO_PRETRAINED_MODEL'
JOB_DIR='PATH_TO_SAVE_CHECKPOINTS'

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)
python submit_lp.py \
        --job_dir ${JOB_DIR} \
        --nodes 4 \
        --model vit_base_patch16_224 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --data_set UCF101 \
        --nb_classes 101 \
        --batch_size 8 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 16 \
        --sampling_rate 4 \
        --num_sample 2 \
        --opt sgd \
        --lr 1e-1 \
        --warmup_lr 1e-3 \
        --momentum 0.9 \
        --weight_decay 0.0 \
        --epochs 100 \
        --test_num_segment 5 \
        --test_num_crop 3 \
        --fc_drop_rate 0.5 \
        --drop_path 0.2 \
        --use_checkpoint \
        --dist_eval \
        --enable_deepspeed
