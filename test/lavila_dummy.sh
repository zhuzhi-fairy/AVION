EXP_PATH=experiments/dummy/
mkdir $EXP_PATH
PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 \
    scripts/main_lavila_pretrain.py \
    --root datasets/clip_dummy/video_320p_15sec/ \
    --root-val datasets/EK100/EK100_320p_15sec_30fps_libx264/ \
    --train-metadata datasets/clip_dummy/train.pkl \
    --use-flash-attn \
    --grad-checkpointing \
    --use-fast-conv1 \
    --batch-size 256 \
    --freeze-temperature \
    --fused-decode-crop \
    --fix-lr \
    --output-dir $EXP_PATH 2>&1 | tee $EXP_PATH/log.txt