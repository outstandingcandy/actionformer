#!/usr/bin/env python3
"""
从 Hugging Face 下载 ActivityNet 数据集
数据集: YimuWang/ActivityNet
"""

import os
import sys
import argparse
import json
from pathlib import Path

def check_and_install_dependencies():
    """检查并安装必要的依赖"""
    try:
        import datasets
        from huggingface_hub import hf_hub_download
        print(f"✅ datasets 已安装 (版本: {datasets.__version__})")
        return True
    except ImportError:
        print("❌ 缺少必要的库，正在安装...")
        import subprocess
        try:
            subprocess.check_call([
                sys.executable, "-m", "pip", "install", 
                "datasets", "huggingface_hub"
            ])
            print("✅ 依赖安装成功")
            return True
        except Exception as e:
            print(f"❌ 安装失败: {e}")
            print("\n请手动安装:")
            print("  pip install datasets huggingface_hub")
            return False

def download_activitynet_dataset(
    output_dir="./data/activitynet_hf",
    cache_dir=None,
    split="train",
    streaming=False
):
    """
    下载 ActivityNet 数据集
    
    Args:
        output_dir: 输出目录
        cache_dir: 缓存目录
        split: 数据集划分 (train, validation, test)
        streaming: 是否使用流式加载
    """
    from datasets import load_dataset
    
    print("="*80)
    print("开始下载 ActivityNet 数据集")
    print("="*80)
    print(f"数据集: YimuWang/ActivityNet")
    print(f"输出目录: {output_dir}")
    print(f"数据集划分: {split}")
    print(f"流式加载: {streaming}")
    if cache_dir:
        print(f"缓存目录: {cache_dir}")
    print("="*80)
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 加载数据集
        print("\n正在加载数据集...")
        dataset = load_dataset(
            "YimuWang/ActivityNet",
            split=split,
            cache_dir=cache_dir,
            streaming=streaming
        )
        
        print(f"✅ 数据集加载成功!")
        
        if not streaming:
            print(f"\n数据集信息:")
            print(f"  样本数量: {len(dataset)}")
            print(f"  特征字段: {dataset.column_names}")
            
            # 显示第一个样本
            if len(dataset) > 0:
                print(f"\n第一个样本示例:")
                first_sample = dataset[0]
                for key, value in first_sample.items():
                    if isinstance(value, (str, int, float, bool)):
                        print(f"    {key}: {value}")
                    elif isinstance(value, list) and len(value) > 0:
                        print(f"    {key}: [{type(value[0]).__name__} × {len(value)}]")
                    else:
                        print(f"    {key}: {type(value).__name__}")
            
            # 保存数据集
            print(f"\n正在保存数据集到 {output_dir}...")
            
            # 保存为 JSON 格式
            json_file = os.path.join(output_dir, f"{split}_annotations.json")
            dataset_dict = []
            
            print("正在转换数据格式...")
            for i, sample in enumerate(dataset):
                dataset_dict.append(sample)
                if (i + 1) % 100 == 0:
                    print(f"  已处理: {i+1}/{len(dataset)} 样本")
            
            with open(json_file, 'w', encoding='utf-8') as f:
                json.dump(dataset_dict, f, indent=2, ensure_ascii=False)
            print(f"✅ 已保存到: {json_file}")
            
            # 保存为原生格式
            print("\n正在保存为 Arrow 格式...")
            arrow_dir = os.path.join(output_dir, f"{split}_dataset")
            dataset.save_to_disk(arrow_dir)
            print(f"✅ 已保存到: {arrow_dir}")
            
            # 生成统计信息
            print("\n生成数据统计...")
            stats = generate_statistics(dataset, split)
            stats_file = os.path.join(output_dir, f"{split}_statistics.json")
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(stats, f, indent=2, ensure_ascii=False)
            print(f"✅ 统计信息已保存到: {stats_file}")
            
        else:
            print("\n流式加载模式，数据将在使用时下载")
            print("示例使用:")
            print("  for sample in dataset:")
            print("      print(sample)")
        
        return dataset
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def generate_statistics(dataset, split):
    """生成数据集统计信息"""
    stats = {
        "split": split,
        "total_samples": len(dataset),
        "column_names": dataset.column_names,
    }
    
    # 统计每个字段的信息
    if len(dataset) > 0:
        first_sample = dataset[0]
        field_types = {}
        for key, value in first_sample.items():
            field_types[key] = type(value).__name__
        stats["field_types"] = field_types
    
    return stats

def convert_to_actionformer_format(
    hf_dataset_dir,
    output_annotation_file,
    split="train"
):
    """
    将 Hugging Face 数据集转换为 ActionFormer 格式
    
    Args:
        hf_dataset_dir: HF 数据集目录
        output_annotation_file: 输出标注文件路径
        split: 数据集划分
    """
    from datasets import load_from_disk
    
    print("\n" + "="*80)
    print("转换为 ActionFormer 格式")
    print("="*80)
    
    # 加载数据集
    dataset_path = os.path.join(hf_dataset_dir, f"{split}_dataset")
    if not os.path.exists(dataset_path):
        print(f"❌ 数据集目录不存在: {dataset_path}")
        return False
    
    print(f"加载数据集: {dataset_path}")
    dataset = load_from_disk(dataset_path)
    
    # 转换格式
    print("正在转换格式...")
    actionformer_data = {
        "version": "1.3",
        "database": {}
    }
    
    for sample in dataset:
        # 根据实际数据集结构调整这里的字段映射
        video_id = sample.get('video_id', sample.get('id', ''))
        
        video_data = {
            "subset": split,
            "duration": sample.get('duration', 0),
            "annotations": []
        }
        
        # 添加 FPS（如果有）
        if 'fps' in sample:
            video_data['fps'] = sample['fps']
        
        # 添加标注
        if 'annotations' in sample:
            for ann in sample['annotations']:
                video_data['annotations'].append({
                    "segment": ann.get('segment', [0, 0]),
                    "label": ann.get('label', ''),
                    "label_id": ann.get('label_id', 0)
                })
        
        actionformer_data['database'][video_id] = video_data
    
    # 保存
    os.makedirs(os.path.dirname(output_annotation_file), exist_ok=True)
    with open(output_annotation_file, 'w', encoding='utf-8') as f:
        json.dump(actionformer_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ 已转换并保存到: {output_annotation_file}")
    print(f"   视频数量: {len(actionformer_data['database'])}")
    
    return True

def download_all_splits(output_dir, cache_dir=None):
    """下载所有数据集划分"""
    splits = ["train", "validation"]
    
    print("\n" + "="*80)
    print("下载所有数据集划分")
    print("="*80)
    
    for split in splits:
        print(f"\n{'='*80}")
        print(f"下载 {split} 集")
        print(f"{'='*80}")
        
        dataset = download_activitynet_dataset(
            output_dir=output_dir,
            cache_dir=cache_dir,
            split=split,
            streaming=False
        )
        
        if dataset is None:
            print(f"❌ {split} 集下载失败")
            continue
        
        print(f"✅ {split} 集下载完成\n")

def main():
    parser = argparse.ArgumentParser(
        description='从 Hugging Face 下载 ActivityNet 数据集'
    )
    parser.add_argument('--output-dir', type=str, 
                        default='./data/activitynet_hf',
                        help='输出目录')
    parser.add_argument('--cache-dir', type=str, default=None,
                        help='缓存目录 (默认使用 HF 默认缓存)')
    parser.add_argument('--split', type=str, default='train',
                        choices=['train', 'validation', 'test', 'all'],
                        help='数据集划分')
    parser.add_argument('--streaming', action='store_true',
                        help='使用流式加载（不完整下载）')
    parser.add_argument('--convert', action='store_true',
                        help='转换为 ActionFormer 格式')
    parser.add_argument('--check-only', action='store_true',
                        help='仅检查依赖，不下载')
    
    args = parser.parse_args()
    
    # 检查依赖
    print("\n检查依赖...")
    if not check_and_install_dependencies():
        return
    
    if args.check_only:
        print("\n✅ 依赖检查完成")
        return
    
    # 下载数据集
    if args.split == 'all':
        download_all_splits(args.output_dir, args.cache_dir)
    else:
        dataset = download_activitynet_dataset(
            output_dir=args.output_dir,
            cache_dir=args.cache_dir,
            split=args.split,
            streaming=args.streaming
        )
        
        if dataset is None:
            print("\n❌ 下载失败")
            return
    
    # 转换格式
    if args.convert and not args.streaming:
        if args.split == 'all':
            for split in ['train', 'validation']:
                convert_to_actionformer_format(
                    args.output_dir,
                    os.path.join(args.output_dir, f'anet_{split}.json'),
                    split=split
                )
        else:
            convert_to_actionformer_format(
                args.output_dir,
                os.path.join(args.output_dir, f'anet_{args.split}.json'),
                split=args.split
            )
    
    print("\n" + "="*80)
    print("✅ 所有操作完成!")
    print("="*80)
    print(f"\n数据保存在: {args.output_dir}")
    print("\n后续步骤:")
    print("1. 检查下载的数据")
    print(f"   ls -lh {args.output_dir}")
    print("2. 查看统计信息")
    print(f"   cat {args.output_dir}/train_statistics.json")
    print("3. 如需转换格式，运行:")
    print(f"   python3 {sys.argv[0]} --split train --convert")

if __name__ == '__main__':
    main()

