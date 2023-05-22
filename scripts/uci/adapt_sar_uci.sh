target=(0 1 2 3 4)

for target_index in ${target[@]}
    do
    python adapt.py \
    --target_domain $target_index \
    --dataset_cfg './cfg/dataset/uci.yaml' \
    --algorithm_cfg './cfg/algorithm/sar.yaml'
done