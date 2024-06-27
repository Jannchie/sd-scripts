DS_PATH='/mnt/nfs-mnj-hot-09/tmp/yande'
python ./finetune/tag_by_wdtagger.py --full_path $DS_PATH
python ./finetune/merge_all_to_metadata.py --full_path $DS_PATH  $DS_PATH/meta.json
python ./check_files.py $DS_PATH/meta.json

DS_PATH='/mnt/nfs-mnj-hot-09/tmp/yande/000'
python ./finetune/prepare_buckets_latents_1.py \
    --batch_size=16 \
    --recursive \
    --full_path \
    --bucket_reso_steps=64 \
    --skip_existing \
    --batch_size=16 \
    --mixed_precision=fp16 \
    $DS_PATH \
    $DS_PATH/meta.json \
    $DS_PATH/meta.json \
    'madebyollin/sdxl-vae-fp16-fix'

accelerate launch ./sdxl_train.py --config_file="./config.toml"
