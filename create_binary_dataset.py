"""
创建二分类数据集：发球 vs 非发球
将原有的12类动作合并为2类
"""
import json
import os
import shutil
from pathlib import Path

def create_binary_badminton_dataset():
    """创建二分类羽毛球数据集"""
    
    print("=" * 100)
    print("创建二分类羽毛球数据集：发球 vs 非发球")
    print("=" * 100)
    
    # 输入和输出路径
    input_dir = "data/actionformer_badminton_swinv2_384_25fps"
    output_dir = "data/actionformer_badminton_binary_25fps"
    
    # 动作映射：发球=0，非发球=1
    action_mapping = {
        "unknown": 1,      # 非发球
        "勾球": 1,         # 非发球
        "发球": 0,         # 发球
        "吊球": 1,         # 非发球
        "扑球": 1,         # 非发球
        "抽球": 1,         # 非发球
        "挑球": 1,         # 非发球
        "挡网": 1,         # 非发球
        "推球": 1,         # 非发球
        "放网前球": 1,     # 非发球
        "杀球": 1,         # 非发球
        "高远球": 1        # 非发球
    }
    
    # 二分类标签映射
    binary_labels = {
        0: "发球",
        1: "非发球"
    }
    
    print(f"\n【动作映射】")
    print(f"发球 (0): 发球")
    print(f"非发球 (1): 其他所有动作")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(os.path.join(output_dir, "features"), exist_ok=True)
    
    # 1. 复制特征文件
    print(f"\n【复制特征文件】")
    input_feat_dir = os.path.join(input_dir, "features")
    output_feat_dir = os.path.join(output_dir, "features")
    
    if os.path.exists(input_feat_dir):
        for feat_file in os.listdir(input_feat_dir):
            if feat_file.endswith('.npy'):
                src = os.path.join(input_feat_dir, feat_file)
                dst = os.path.join(output_feat_dir, feat_file)
                shutil.copy2(src, dst)
        print(f"✓ 已复制特征文件到: {output_feat_dir}")
    else:
        print(f"✗ 特征目录不存在: {input_feat_dir}")
        return
    
    # 2. 处理标注文件
    print(f"\n【处理标注文件】")
    input_annotations = os.path.join(input_dir, "badminton_annotations.json")
    output_annotations = os.path.join(output_dir, "badminton_annotations.json")
    
    if not os.path.exists(input_annotations):
        print(f"✗ 标注文件不存在: {input_annotations}")
        return
    
    with open(input_annotations, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 统计原始动作分布
    original_action_counts = {}
    binary_action_counts = {0: 0, 1: 0}
    
    # 转换标注
    for video_id, video_data in data.items():
        if 'segments' in video_data and 'labels' in video_data:
            new_segments = []
            new_labels = []
            
            for segment, label in zip(video_data['segments'], video_data['labels']):
                # 获取原始动作名称
                original_action = None
                for action_name, action_id in data.get('action_mapping', {}).get('action_to_id', {}).items():
                    if action_id == label:
                        original_action = action_name
                        break
                
                if original_action is None:
                    # 如果没有找到映射，尝试从label直接获取
                    action_names = ["unknown", "勾球", "发球", "吊球", "扑球", "抽球", "挑球", "挡网", "推球", "放网前球", "杀球", "高远球"]
                    if 0 <= label < len(action_names):
                        original_action = action_names[label]
                
                if original_action is not None:
                    # 统计原始动作
                    original_action_counts[original_action] = original_action_counts.get(original_action, 0) + 1
                    
                    # 转换为二分类标签
                    binary_label = action_mapping.get(original_action, 1)  # 默认为非发球
                    binary_action_counts[binary_label] += 1
                    
                    new_segments.append(segment)
                    new_labels.append(binary_label)
            
            # 更新视频数据
            video_data['segments'] = new_segments
            video_data['labels'] = new_labels
    
    # 保存转换后的标注文件
    with open(output_annotations, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已保存二分类标注文件: {output_annotations}")
    
    # 3. 创建动作映射文件
    print(f"\n【创建动作映射文件】")
    action_mapping_file = os.path.join(output_dir, "action_mapping.json")
    
    action_mapping_data = {
        "action_to_id": {
            "发球": 0,
            "非发球": 1
        },
        "id_to_action": {
            "0": "发球",
            "1": "非发球"
        }
    }
    
    with open(action_mapping_file, 'w', encoding='utf-8') as f:
        json.dump(action_mapping_data, f, ensure_ascii=False, indent=2)
    
    print(f"✓ 已保存动作映射文件: {action_mapping_file}")
    
    # 4. 复制分割文件
    print(f"\n【复制分割文件】")
    input_splits = os.path.join(input_dir, "splits.json")
    output_splits = os.path.join(output_dir, "splits.json")
    
    if os.path.exists(input_splits):
        shutil.copy2(input_splits, output_splits)
        print(f"✓ 已复制分割文件: {output_splits}")
    else:
        print(f"✗ 分割文件不存在: {input_splits}")
    
    # 5. 统计信息
    print(f"\n" + "=" * 100)
    print("转换统计")
    print("=" * 100)
    
    print(f"\n【原始动作分布】")
    for action, count in sorted(original_action_counts.items()):
        print(f"  {action}: {count}个")
    
    print(f"\n【二分类动作分布】")
    for label_id, count in binary_action_counts.items():
        action_name = binary_labels[label_id]
        print(f"  {action_name} (ID={label_id}): {count}个")
    
    total_actions = sum(binary_action_counts.values())
    serve_ratio = binary_action_counts[0] / total_actions * 100
    non_serve_ratio = binary_action_counts[1] / total_actions * 100
    
    print(f"\n【类别平衡性】")
    print(f"  发球: {binary_action_counts[0]}个 ({serve_ratio:.1f}%)")
    print(f"  非发球: {binary_action_counts[1]}个 ({non_serve_ratio:.1f}%)")
    print(f"  总计: {total_actions}个")
    
    if serve_ratio < 10 or serve_ratio > 90:
        print(f"  ⚠️  类别不平衡！发球占比{serve_ratio:.1f}%")
    else:
        print(f"  ✓ 类别相对平衡")
    
    # 6. 创建配置文件
    print(f"\n【创建配置文件】")
    config_file = "configs/badminton_binary_25fps.yaml"
    
    config_content = f"""# 二分类羽毛球数据集配置 (发球 vs 非发球)
dataset_name: "badminton_new"
train_split: ['train']
val_split: ['val']

# 数据路径配置
dataset:
  json_file: "{output_annotations}"
  feat_folder: "{output_feat_dir}"
  file_ext: ".npy"
  
  # 数据分割配置
  split_file: "{output_splits}"
  
  # 特征配置
  feat_stride: 1
  num_frames: 16
  default_fps: 25.0
  feat_offset: 8.0
  
  # 序列长度限制
  max_seq_len: 256  # 增加以支持更长视频
  
  # 数据增强
  input_dim: 1024  # SwinV2-Base 384特征维度
  num_classes: 2   # 二分类：发球(0) vs 非发球(1)
  
  # 其他配置
  remove_duplicate: false
  trim_method: "truncate"
  force_upsampling: false
  trunc_thresh: 0.5

# model config
model_name: "LocPointTransformer"
model:
  # 基础配置
  input_dim: 1024  # SwinV2-Base 384特征维度
  num_classes: 2   # 二分类
  max_seq_len: 256
  
  # 训练配置
  train_cfg:
    # 损失配置
    init_loss_norm: 1000
    loss_normalizer_momentum: 0.9
    loss_weight: 0.0  # 自动计算
    
    # 分类配置
    cls_prior_prob: 0.1  # 提高正样本先验概率
    label_smoothing: 0.1
    
    # 中心采样配置
    center_sample: radius
    center_sample_radius: 1.5
    
    # 正则化
    dropout: 0.1
    droppath: 0.1
    
    # 空类别
    head_empty_cls: []
  
  # 测试配置
  test_cfg:
    pre_nms_thresh: 0.001
    pre_nms_topk: 5000
    iou_threshold: 0.9
    min_score: 0.001
    max_seg_num: 2000
    nms_method: soft
    duration_thresh: 0.05
    multiclass_nms: True
    nms_sigma: 0.9
    voting_thresh: 0.95

# 网络架构配置
backbone_type: convTransformer
fpn_type: identity
backbone_arch: [2, 2, 5]
scale_factor: 2
max_buffer_len_factor: 6.0
n_head: 4
n_mha_win_size: -1
embd_kernel_size: 3
embd_dim: 512
embd_with_ln: True
fpn_dim: 512
fpn_with_ln: True
fpn_start_level: 0
head_dim: 512
regression_range: [[0, 4], [4, 8], [8, 16], [16, 32], [32, 64], [64, 10000]]
head_num_layers: 3
head_kernel_size: 3
head_with_ln: True
use_abs_pe: True
use_rel_pe: False

# 优化器配置
opt:
  learning_rate: 0.0001
  epochs: 160
  weight_decay: 0.05
  warmup_epochs: 20  # 增加warmup
  lr_schedule: cosine
  clip_grad_l2norm: 1.0

# 数据加载器配置
loader:
  batch_size: 4
  num_workers: 4
  pin_memory: True
  drop_last: True

# 训练评估配置
train_eval_cfg:
  eval_freq: 5
  save_detailed: true
  max_batches: 2  # 只评估前2个batch

# 输出配置
output_folder: "./ckpt/"
print_freq: 1
"""
    
    with open(config_file, 'w', encoding='utf-8') as f:
        f.write(config_content)
    
    print(f"✓ 已创建配置文件: {config_file}")
    
    print(f"\n" + "=" * 100)
    print("二分类数据集创建完成！")
    print("=" * 100)
    
    print(f"""
【输出目录】: {output_dir}
【配置文件】: {config_file}

【数据集特点】:
  - 类别数: 2 (发球 vs 非发球)
  - 特征: SwinV2-384, 25 FPS
  - 最大序列长度: 256帧
  - 类别分布: 发球{serve_ratio:.1f}% vs 非发球{non_serve_ratio:.1f}%

【训练命令】:
  python train_with_eval.py {config_file} --output binary_exp

【预期效果】:
  - 二分类任务相对简单
  - 发球动作特征明显
  - 预期mAP@0.5 > 0.3
""")

if __name__ == "__main__":
    create_binary_badminton_dataset()
