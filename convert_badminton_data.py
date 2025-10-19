#!/usr/bin/env python3
"""
将羽毛球数据转换为ActionFormer格式
"""

import os
import sys
import json
import numpy as np
import argparse
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def convert_badminton_to_actionformer(input_json, output_dir, video_dir, binary_serve=False):
    """
    将羽毛球数据转换为ActionFormer格式
    
    Args:
        input_json: 输入JSON文件路径
        output_dir: 输出目录
        video_dir: 视频文件目录
        binary_serve: 是否使用二分类模式（发球 vs 非发球）
    """
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 读取原始数据
    print(f"读取数据文件: {input_json}")
    with open(input_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    print(f"总视频数: {len(data)}")
    
    if binary_serve:
        print("\n🎯 使用发球检测模式: 只保留发球标注")
        # 发球检测模式：只保留发球，忽略其他动作
        all_hit_types = {'发球'}
    else:
        # 动作类型映射
        hit_type_mapping = {
            'serve': '发球',
            'push shot': '推球', 
            'drop shot': '吊球',
            'clear': '高远球',
            'kill': '杀球',
            'net shot': '放网前球',
            'net kill': '扑球',
            'net lift': '挑球',
            'drive': '抽球',
            'block': '挡网',
            'cross-court net shot': '勾球',
            '': 'unknown'  # 空字符串映射为unknown
        }
        
        # 收集所有动作类型（包含hitter信息）
        all_hit_types = set()
        for item in data:
            for hit in item.get('hitting', []):
                hit_type = hit.get('hit_type', '')
                hitter = hit.get('hitter', 'unknown')  # 获取击球者位置
                
                # 映射动作类型
                if hit_type in hit_type_mapping:
                    mapped_type = hit_type_mapping[hit_type]
                else:
                    mapped_type = 'unknown'
                
                # 组合动作类型和击球者位置: "动作_位置"
                combined_label = f"{mapped_type}_{hitter}"
                all_hit_types.add(combined_label)
    
    # 创建动作到ID的映射
    action_list = sorted(list(all_hit_types))
    action_to_id = {action: idx for idx, action in enumerate(action_list)}
    
    print(f"动作类型: {action_list}")
    print(f"动作数量: {len(action_list)}")
    
    # 转换数据
    actionformer_db = {}
    valid_videos = 0
    
    for item in tqdm(data, desc="转换视频数据"):
        video_file = item.get('video', '')
        if not video_file:
            continue
            
        video_path = os.path.join(video_dir, video_file)
        if not os.path.exists(video_path):
            print(f"警告: 视频文件不存在: {video_path}")
            continue
        
        # 生成视频ID (去掉.mp4扩展名)
        video_id = os.path.splitext(video_file)[0]
        
        # 获取视频基本信息
        fps = item.get('fps', 25.0)
        duration_frames = item.get('duration_frames', 0)
        start_frame = item.get('start_frame', 0)
        end_frame = item.get('end_frame', 0)
        
        # 计算视频时长 (秒)
        duration = duration_frames / fps if fps > 0 else 0
        
        # 解析动作标注
        segments = []
        labels = []
        
        for hit in item.get('hitting', []):
            hit_start_frame = hit.get('start_frame', 0)
            hit_end_frame = hit.get('end_frame', 0)
            hit_type = hit.get('hit_type', '')
            hitter = hit.get('hitter', 'unknown')  # 获取击球者位置
            
            # 跳过无效的动作
            if hit_start_frame >= hit_end_frame:
                continue
                
            # 转换为相对时间 (相对于视频开始)
            start_time = (hit_start_frame - start_frame) / fps
            end_time = (hit_end_frame - start_frame) / fps
            
            # 确保时间在视频范围内
            if start_time < 0 or end_time > duration:
                continue
            
            # 根据模式确定标签
            if binary_serve:
                # 发球检测模式：只保留发球，跳过其他动作
                if hit_type == 'serve':
                    combined_label = '发球'
                else:
                    # 跳过非发球动作
                    continue
            else:
                # 多分类模式：映射动作类型
                if hit_type in hit_type_mapping:
                    mapped_type = hit_type_mapping[hit_type]
                else:
                    mapped_type = 'unknown'
                
                # 组合动作类型和击球者位置: "动作_位置"
                combined_label = f"{mapped_type}_{hitter}"
            
            segments.append([start_time, end_time])
            labels.append(action_to_id[combined_label])
        
        # 只保存有动作标注的视频
        if segments and labels:
            actionformer_db[video_id] = {
                'duration': duration,
                'segments': segments,
                'labels': labels
            }
            valid_videos += 1
        else:
            print(f"跳过无动作标注的视频: {video_id}")
    
    print(f"有效视频数: {valid_videos}/{len(data)}")
    
    # 保存ActionFormer格式的标注文件
    output_json = os.path.join(output_dir, 'badminton_annotations.json')
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(actionformer_db, f, indent=2, ensure_ascii=False)
    
    # 保存动作映射
    mapping_file = os.path.join(output_dir, 'action_mapping.json')
    mapping_data = {
        'actions': action_list,
        'action_to_id': action_to_id,
        'binary_serve': binary_serve
    }
    if not binary_serve:
        mapping_data['hit_type_mapping'] = hit_type_mapping
    
    with open(mapping_file, 'w', encoding='utf-8') as f:
        json.dump(mapping_data, f, indent=2, ensure_ascii=False)
    
    # 创建训练/验证分割
    video_ids = list(actionformer_db.keys())
    np.random.seed(42)  # 固定随机种子
    np.random.shuffle(video_ids)
    
    # 80% 训练，20% 验证
    split_idx = int(len(video_ids) * 0.8)
    train_videos = video_ids[:split_idx]
    val_videos = video_ids[split_idx:]
    
    # 保存分割信息
    split_file = os.path.join(output_dir, 'splits.json')
    with open(split_file, 'w', encoding='utf-8') as f:
        json.dump({
            'train': train_videos,
            'val': val_videos
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\n转换完成!")
    print(f"输出目录: {output_dir}")
    print(f"标注文件: {output_json}")
    print(f"动作映射: {mapping_file}")
    print(f"数据分割: {split_file}")
    print(f"训练视频: {len(train_videos)}")
    print(f"验证视频: {len(val_videos)}")
    
    return actionformer_db, action_to_id

def main():
    parser = argparse.ArgumentParser(description='转换羽毛球数据为ActionFormer格式')
    parser.add_argument('--input_json', type=str, 
                        default='data/transformed_combined_rounds_output_en_evals_translated.json',
                        help='输入JSON文件路径')
    parser.add_argument('--output_dir', type=str, 
                        default='data/actionformer_badminton_new',
                        help='输出目录')
    parser.add_argument('--video_dir', type=str, 
                        default='data/videos',
                        help='视频文件目录')
    parser.add_argument('--binary_serve', action='store_true',
                        help='使用发球检测模式：只保留发球标注（默认：多分类模式）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.input_json):
        print(f"错误: 输入文件不存在: {args.input_json}")
        return
    
    if not os.path.exists(args.video_dir):
        print(f"错误: 视频目录不存在: {args.video_dir}")
        return
    
    # 如果是二分类模式，自动调整输出目录
    if args.binary_serve and args.output_dir == 'data/actionformer_badminton_new':
        args.output_dir = 'data/actionformer_badminton_binary_serve'
        print(f"📁 二分类模式，输出目录自动设置为: {args.output_dir}")
    
    # 转换数据
    convert_badminton_to_actionformer(
        args.input_json, 
        args.output_dir, 
        args.video_dir,
        binary_serve=args.binary_serve
    )

if __name__ == '__main__':
    main()
