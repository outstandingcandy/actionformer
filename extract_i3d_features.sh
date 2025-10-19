#!/bin/bash

# I3D特征提取示例脚本
# 作者: 自动生成
# 日期: 2025-10-14

# ============================================
# 1. 羽毛球数据集 - I3D特征提取
# ============================================
echo "提取羽毛球数据集的I3D特征..."
python3 extract_features.py \
    --annotations_file /data/badminton/actionformer_badminton_binary_serve/badminton_annotations.json \
    --video_dir /data/badminton/videos \
    --output_dir /data/badminton/actionformer_badminton_i3d \
    --feature_type i3d \
    --dataset_format badminton \
    --target_fps 25.0 \
    --i3d_frame_count 32

# ============================================
# 2. ActivityNet数据集 - I3D特征提取（验证集）
# ============================================
# echo "提取ActivityNet验证集的I3D特征..."
# python3 extract_features.py \
#     --annotations_file /data/activitynet/activity_net.v1-3.min.json \
#     --video_dir /data/activitynet/videos \
#     --output_dir /data/activitynet/activitynet_i3d \
#     --feature_type i3d \
#     --dataset_format activitynet \
#     --subset validation \
#     --target_fps 25.0 \
#     --i3d_frame_count 32

echo "I3D特征提取完成！"


