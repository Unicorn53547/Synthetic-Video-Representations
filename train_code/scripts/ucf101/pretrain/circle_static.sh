export NCCL_DEBUG=INFO
# Set the path to save checkpoints
OUTPUT_DIR=''
# Set the path to HMDB51 train set. 
DATA_PATH='Annotations/YOUR_HMDB_PATH'
#PATH TO MIXED IMGDATASET
IMG_PATH='PATH_TO_MIXED_IMGDATASET'
IMG_META=''
JOB_DIR='PATH_TO_SAVE_CHECKPOINTS'

# batch_size can be adjusted according to number of GPUs
# this script is for 32 GPUs (4 nodes x 8 GPUs)
python submit_pretrain.py \
        --job_dir ${JOB_DIR} \
        --nodes 4 \
        --data_path ${DATA_PATH} \
        --mask_type tube \
        --mask_ratio 0.75 \
        --model pretrain_videomae_base_patch16_224 \
        --decoder_depth 4 \
        --batch_size 6 \
        --accum_iter 1 \
        --num_frames 16 \
        --sampling_rate 4 \
        --opt adamw \
        --lr 3e-4 \
        --opt_betas 0.9 0.95 \
        --warmup_epochs 40 \
        --save_ckpt_freq 800 \
        --epochs 3201 \
        --log_dir ${OUTPUT_DIR} \
        --output_dir ${OUTPUT_DIR} \
        --shape circle \
        --virtual_dataset_size 9537 \
        --max_iters 500 \
        --max_acc 0.03 \
        --min_speed 1.2 \
        --max_speed 1.5 \
        --mixed \
        --static \
        --img_data_type imagenet \
        --img_data_path ${IMG_PATH} \
        --metadata ${IMG_META} \
        --use_real_ratio 1.0 \
        --mixed_ratio 0.0
        # --no_pin_mem
