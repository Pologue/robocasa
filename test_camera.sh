#!/bin/bash

conda activate robocasa

# 定义 camera_name 列表
cameras=(
    "robot0_agentview_center"
    "robot0_agentview_left"
    "robot0_agentview_right"
    "robot0_frontview"
    # "robot0_robotview"
    "robot0_eye_in_hand"
)

dataset=datasets/v1.0/pretrain/composite/PrepareCoffee/20250716/lerobot
# dataset=datasets/v1.0/pretrain/composite/PackIdenticalLunches/20250805/lerobot
# dataset=datasets/v1.0/pretrain/composite/DeliverStraw/20250723/lerobot
# dataset=datasets/v1.0/pretrain/atomic/PickPlaceCounterToCabinet/20250819/lerobot

task_dir=$(dirname "$(dirname "$dataset")")
task_name=$(basename "$task_dir")

# 遍历每个 camera_name 并运行命令
for cam in "${cameras[@]}"; do
    echo "Running for camera: $cam"
    python -m robocasa.scripts.dataset_scripts.extract_affordance_gt \
        --dataset "${dataset}" \
        --output "outputs/${task_name}/${cam}/affordance_summary_masked.jsonl" \
        --n 1 \
        --save_dense \
        --dense_dir "outputs/${task_name}/${cam}/dense_masked" \
        --dense_format npz \
        --debug_video_dir "outputs/${task_name}/${cam}/debug_videos_masked" \
        --debug_video_max_episodes 1 \
        --enforce_expected_subtasks \
        --verbose \
        --camera_name "$cam"
done