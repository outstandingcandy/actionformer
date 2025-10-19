#!/usr/bin/env python3
"""
从 Hugging Face 下载 ActivityNet 数据集 (简化版)
数据集: YimuWang/ActivityNet

注意: 这个数据集似乎只包含少量视频样本
"""

import os
from huggingface_hub import hf_hub_download, list_repo_files, snapshot_download
import argparse

def download_dataset_files(repo_id="YimuWang/ActivityNet", output_dir="./data/activitynet_hf"):
    """
    下载数据集的所有文件
    """
    print("="*80)
    print(f"从 Hugging Face 下载数据集: {repo_id}")
    print("="*80)
    
    try:
        # 列出所有文件
        print("\n列出仓库文件...")
        files = list_repo_files(repo_id, repo_type="dataset")
        
        print(f"\n仓库中的文件 (共 {len(files)} 个):")
        for f in files:
            print(f"  - {f}")
        
        # 下载整个仓库
        print(f"\n下载所有文件到: {output_dir}")
        local_dir = snapshot_download(
            repo_id=repo_id,
            repo_type="dataset",
            local_dir=output_dir,
            local_dir_use_symlinks=False
        )
        
        print(f"\n✅ 下载完成!")
        print(f"文件保存在: {local_dir}")
        
        # 列出下载的文件
        print(f"\n下载的文件:")
        for root, dirs, files in os.walk(local_dir):
            for file in files:
                file_path = os.path.join(root, file)
                file_size = os.path.getsize(file_path)
                rel_path = os.path.relpath(file_path, local_dir)
                print(f"  {rel_path} ({file_size/1024/1024:.2f} MB)")
        
        return local_dir
        
    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def load_dataset_without_decode(repo_id="YimuWang/ActivityNet", output_dir="./data/activitynet_hf"):
    """
    加载数据集但不解码视频
    """
    from datasets import load_dataset
    
    print("\n" + "="*80)
    print("加载数据集（禁用视频解码）")
    print("="*80)
    
    try:
        # 尝试加载但不自动解码
        print("\n正在加载数据集...")
        
        # 使用 with_format 来避免自动解码
        dataset = load_dataset(
            repo_id,
            split='train',
            cache_dir=output_dir
        )
        
        # 禁用视频解码
        dataset = dataset.cast_column('video', datasets.Video(decode=False))
        
        print(f"✅ 数据集加载成功!")
        print(f"  样本数量: {len(dataset)}")
        
        # 显示样本信息
        print(f"\n前3个样本:")
        for i in range(min(3, len(dataset))):
            sample = dataset[i]
            print(f"\n样本 {i+1}:")
            for key, value in sample.items():
                if isinstance(value, dict) and 'path' in value:
                    print(f"  {key}: {value['path']}")
                elif isinstance(value, (str, int, float)):
                    print(f"  {key}: {value}")
                else:
                    print(f"  {key}: {type(value).__name__}")
        
        return dataset
        
    except Exception as e:
        print(f"\n视频解码方式失败: {e}")
        print("建议直接下载文件而不是使用 datasets 库")
        return None

def main():
    parser = argparse.ArgumentParser(
        description='从 Hugging Face 下载 ActivityNet 数据集（简化版）'
    )
    parser.add_argument('--repo-id', type=str, 
                        default='YimuWang/ActivityNet',
                        help='Hugging Face 数据集 ID')
    parser.add_argument('--output-dir', type=str,
                        default='./data/activitynet_hf',
                        help='输出目录')
    parser.add_argument('--method', type=str,
                        default='files',
                        choices=['files', 'dataset'],
                        help='下载方法: files(直接下载文件) 或 dataset(使用datasets库)')
    
    args = parser.parse_args()
    
    print(f"\nHugging Face 数据集: {args.repo_id}")
    print(f"输出目录: {args.output_dir}")
    print(f"下载方法: {args.method}")
    
    if args.method == 'files':
        # 直接下载文件
        result = download_dataset_files(args.repo_id, args.output_dir)
        
        if result:
            print("\n" + "="*80)
            print("✅ 下载完成!")
            print("="*80)
            print(f"\n文件位置: {result}")
            print("\n您可以:")
            print(f"  1. 查看文件: ls -lh {result}")
            print(f"  2. 查看视频文件:")
            print(f"     find {result} -name '*.mp4' -o -name '*.avi'")
    else:
        # 使用 datasets 库
        result = load_dataset_without_decode(args.repo_id, args.output_dir)
        
        if result:
            print("\n数据集加载成功!")

if __name__ == '__main__':
    main()

