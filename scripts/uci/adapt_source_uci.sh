target=(0 1 2 3 4)

for target_index in ${target[@]}
    do
    python adapt.py \
    --target_domain $target_index \
    --dataset_cfg './cfg/dataset/uci.yaml' \
    --algorithm_cfg './cfg/algorithm/source.yaml'
done


# python adapt.py \
# --target_domain S1 \
# --dataset_cfg './cfg/dataset/oppo.yaml'

# python adapt.py \
# --target_domain S2 \
# --cfg './cfg/dataset/oppo.yaml'

# python adapt.py \
# --target_domain S3 \
# --cfg './cfg/dataset/oppo.yaml'

# python adapt.py \
# --target_domain S4 \
# --cfg './cfg/dataset/oppo.yaml'
