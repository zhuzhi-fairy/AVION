OUTPUT_DIR='experiments/videomae_pretrain_clip2mae_vitl_hktk555_1200e_2'
mkdir -p $OUTPUT_DIR
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 main_videomae_pretrain_from_clip_vit.py \
    --root datasets/hktk/sss/annofab-data/ \
    --train-metadata datasets/hktk/hktk555.txt \
    --num-samples 215184 \
    --model VIDEOMAE_CLIP_VITL14 \
    --clip-ckpt checkpoints/avion_pretrain_lavila_vitl_best.pt \
    --use-flash-attn-at-encoder \
    --use-flash-attn-at-decoder \
    --fused-decode-crop \
    --channel-last \
    --use-multi-epochs-loader \
    --print-freq 1 \
    -j 8 \
    --drop-path-rate 0.2 \
    --batch-size 64 \
    --epochs 1200 \
    --warmup-epochs 10 \
    --optimizer adamw \
    --lr 1e-5 \
    --lr-start 1e-10 \
    --lr-end 1e-10 \
    --wd 0.05 \
    --betas 0.9 0.95 \
    --save-freq 10 \
    --grad-clip-norm 0.02 \
    --output-dir $OUTPUT_DIR 2>&1 | tee ${OUTPUT_DIR}/log.txt
