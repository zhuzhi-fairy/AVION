mkdir $EXP_PATH
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 scripts/main_lavila_finetune_mir.py \
    --root datasets/EK100/EK100_320p_15sec_30fps_libx264/ \
    --video-chunk-length 15 --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 64 \
    --fused-decode-crop \
    --use-multi-epochs-loader \
    --pretrain-model experiments/pretrain_lavila_vitb/checkpoint_best.pt \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt
