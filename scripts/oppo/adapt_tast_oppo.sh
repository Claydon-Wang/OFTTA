target=(S1 S2 S3 S4)

for target_index in ${target[@]}
    do
    python adapt.py \
    --target_domain $target_index \
    --dataset_cfg './cfg/dataset/oppo.yaml' \
    --algorithm_cfg './cfg/algorithm/tast.yaml'
done
