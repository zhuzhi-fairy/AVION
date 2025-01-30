PYTHONPATH=.:third_party/decord/python/ torchrun \
    --nproc_per_node=8 main_videomae_pretrain.py \
    --root ./sss/annofab-data/ \
    --train-metadata datasets/hktk555_pt/train.txt \
    --num_samples 215184 \
    --model VIDEOMAE_VITL16 \
    --use-flash-attn-at-encoder --use-flash-attn-at-decoder \
    --batch-size 192 --channel-last \
    --fused-decode-crop --use-multi-epochs-loader --optimizer lion \
    -j 8 \
    --output-dir experiments/videomae_pretrain_vitl_lion_hktk555_pt 2>&1 | tee experiments/videomae_pretrain_vitl_lion_hktk555_pt/log.txt
