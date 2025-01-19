# Set the path to save checkpoints
OUTPUT_DIR=''
# Set the path to HMDB51 train set. 
DATA_PATH='Annotations/YOUR_HMDBPATH'
MODEL_PATH='PATH_TO_PRETRAINED_MODEL'
JOB_DIR='PATH_TO_SAVE_CHECKPOINTS'

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)
python submiit_finetune.py \
        --job_dir ${JOB_DIR} \
        --nodes 4 \
        --model vit_base_patch16_224 \
        --data_path ${DATA_PATH} \
        --finetune ${MODEL_PATH} \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --data_set HMDB51 \
        --nb_classes 51 \
        --batch_size 8 \
        --input_size 224 \
        --short_side_size 224 \
        --save_ckpt_freq 100 \
        --num_frames 16 \
        --sampling_rate 2 \
        --num_sample 2 \
        --opt adamw \
        --lr 1e-3 \
        --warmup_lr 1e-6 \
        --min_lr 1e-5 \
        --layer_decay 0.7 \
        --opt_betas 0.9 0.999 \
        --weight_decay 0.05 \
        --epochs 100 \
        --test_num_segment 10 \
        --test_num_crop 3 \
        --fc_drop_rate 0.5 \
        --drop_path 0.2 \
        --use_checkpoint \
        --dist_eval \
        --enable_deepspeed
