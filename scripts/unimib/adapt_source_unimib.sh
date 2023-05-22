target=(1 2 3 5)

for target_index in ${target[@]}
    do
    python adapt.py \
    --target_domain $target_index \
    --dataset_cfg './cfg/dataset/unimib.yaml' \
    --algorithm_cfg './cfg/algorithm/source.yaml'
done