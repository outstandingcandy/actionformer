#!/usr/bin/env python3
"""
选择性下载 ActivityNet 数据集文件
数据集: YimuWang/ActivityNet

支持选择性下载:
- 仅标注文件
- 仅示例视频
- 完整训练/验证/测试集
"""

import os
import argparse
from huggingface_hub import hf_hub_download, list_repo_files
from tqdm import tqdm

REPO_ID = "YimuWang/ActivityNet"

def list_available_files():
    """列出所有可用文件"""
    print("获取文件列表...")
    files = list_repo_files(REPO_ID, repo_type="dataset")
    
    # 分类文件
    categories = {
        'annotations': [],
        'sample_videos': [],
        'train_videos': [],
        'val_videos': [],
        'test_videos': [],
        'charades': [],
        'other': []
    }
    
    for f in files:
        if f.endswith('.json') or f.endswith('.csv') or f.endswith('.txt'):
            if 'Charades' in f:
                categories['charades'].append(f)
            else:
                categories['annotations'].append(f)
        elif 'v1-3/train_val' in f and f.endswith('.mp4'):
            categories['sample_videos'].append(f)
        elif 'train.tar.gz' in f:
            categories['train_videos'].append(f)
        elif 'val.tar.gz' in f:
            categories['val_videos'].append(f)
        elif 'test.tar.gz' in f:
            categories['test_videos'].append(f)
        elif 'Charades' in f:
            categories['charades'].append(f)
        else:
            categories['other'].append(f)
    
    return categories

def print_file_categories(categories):
    """打印文件分类"""
    print("\n" + "="*80)
    print("ActivityNet 数据集文件分类")
    print("="*80)
    
    print(f"\n📝 标注文件 ({len(categories['annotations'])} 个):")
    for f in categories['annotations'][:10]:
        print(f"  - {f}")
    if len(categories['annotations']) > 10:
        print(f"  ... 还有 {len(categories['annotations'])-10} 个文件")
    
    print(f"\n🎬 示例视频 ({len(categories['sample_videos'])} 个):")
    for f in categories['sample_videos']:
        print(f"  - {f}")
    
    print(f"\n📦 训练集视频压缩包 ({len(categories['train_videos'])} 个):")
    print(f"  总共 {len(categories['train_videos'])} 个分卷文件")
    if categories['train_videos']:
        print(f"  示例: {categories['train_videos'][0]}")
    
    print(f"\n📦 验证集视频压缩包 ({len(categories['val_videos'])} 个):")
    print(f"  总共 {len(categories['val_videos'])} 个分卷文件")
    if categories['val_videos']:
        print(f"  示例: {categories['val_videos'][0]}")
    
    print(f"\n📦 测试集视频压缩包 ({len(categories['test_videos'])} 个):")
    print(f"  总共 {len(categories['test_videos'])} 个分卷文件")
    if categories['test_videos']:
        print(f"  示例: {categories['test_videos'][0]}")
    
    print(f"\n🎯 Charades 数据集 ({len(categories['charades'])} 个):")
    print(f"  包含 Charades 数据集的相关文件")
    
    print(f"\n📄 其他文件 ({len(categories['other'])} 个):")
    for f in categories['other']:
        print(f"  - {f}")

def download_files(files_to_download, output_dir):
    """下载指定的文件列表"""
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n开始下载 {len(files_to_download)} 个文件到 {output_dir}")
    print("="*80)
    
    downloaded_files = []
    failed_files = []
    
    for file_path in tqdm(files_to_download, desc="下载进度"):
        try:
            # 计算本地路径
            local_path = os.path.join(output_dir, file_path)
            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            
            # 下载文件
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=file_path,
                local_dir=output_dir,
                local_dir_use_symlinks=False
            )
            
            downloaded_files.append(file_path)
            
        except Exception as e:
            print(f"\n❌ 下载失败: {file_path}")
            print(f"   错误: {e}")
            failed_files.append(file_path)
    
    # 打印摘要
    print("\n" + "="*80)
    print("下载摘要")
    print("="*80)
    print(f"✅ 成功: {len(downloaded_files)} 个文件")
    print(f"❌ 失败: {len(failed_files)} 个文件")
    
    if failed_files:
        print(f"\n失败的文件:")
        for f in failed_files:
            print(f"  - {f}")
    
    return downloaded_files, failed_files

def main():
    parser = argparse.ArgumentParser(
        description='选择性下载 ActivityNet 数据集',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:
  # 列出所有文件
  python3 %(prog)s --list
  
  # 仅下载标注文件
  python3 %(prog)s --annotations
  
  # 下载标注和示例视频
  python3 %(prog)s --annotations --samples
  
  # 下载训练集视频（大文件！）
  python3 %(prog)s --train
  
  # 下载所有内容（非常大！）
  python3 %(prog)s --all
        """
    )
    
    parser.add_argument('--output-dir', type=str,
                        default='./data/activitynet_hf',
                        help='输出目录')
    parser.add_argument('--list', action='store_true',
                        help='仅列出所有可用文件')
    parser.add_argument('--annotations', action='store_true',
                        help='下载标注文件')
    parser.add_argument('--samples', action='store_true',
                        help='下载示例视频（3个视频）')
    parser.add_argument('--train', action='store_true',
                        help='下载训练集视频（大文件！）')
    parser.add_argument('--val', action='store_true',
                        help='下载验证集视频（大文件！）')
    parser.add_argument('--test', action='store_true',
                        help='下载测试集视频（大文件！）')
    parser.add_argument('--all', action='store_true',
                        help='下载所有文件（非常大！）')
    
    args = parser.parse_args()
    
    # 获取文件列表
    categories = list_available_files()
    
    # 打印文件分类
    print_file_categories(categories)
    
    if args.list:
        print("\n使用 --annotations, --samples, --train, --val, --test 等选项下载特定文件")
        return
    
    # 确定要下载的文件
    files_to_download = []
    
    if args.all:
        print("\n⚠️  警告: 将下载所有文件（非常大！）")
        response = input("确认继续？(yes/no): ")
        if response.lower() != 'yes':
            print("取消下载")
            return
        
        # 下载所有文件
        for file_list in categories.values():
            files_to_download.extend(file_list)
    else:
        if args.annotations:
            files_to_download.extend(categories['annotations'])
            print(f"\n✓ 将下载标注文件 ({len(categories['annotations'])} 个)")
        
        if args.samples:
            files_to_download.extend(categories['sample_videos'])
            print(f"✓ 将下载示例视频 ({len(categories['sample_videos'])} 个)")
        
        if args.train:
            files_to_download.extend(categories['train_videos'])
            print(f"✓ 将下载训练集视频 ({len(categories['train_videos'])} 个分卷)")
        
        if args.val:
            files_to_download.extend(categories['val_videos'])
            print(f"✓ 将下载验证集视频 ({len(categories['val_videos'])} 个分卷)")
        
        if args.test:
            files_to_download.extend(categories['test_videos'])
            print(f"✓ 将下载测试集视频 ({len(categories['test_videos'])} 个分卷)")
    
    if not files_to_download:
        print("\n❌ 未选择任何文件下载")
        print("使用 --annotations, --samples, --train, --val, --test 等选项")
        print("或使用 --list 查看所有可用文件")
        return
    
    # 开始下载
    downloaded, failed = download_files(files_to_download, args.output_dir)
    
    print(f"\n文件保存在: {args.output_dir}")
    
    # 显示后续步骤
    if categories['annotations'] and any(f in downloaded for f in categories['annotations']):
        print("\n后续步骤:")
        print("1. 查看标注文件:")
        print(f"   cat {args.output_dir}/activity_net.v1-3.min.json | jq . | head -50")
        
    if categories['sample_videos'] and any(f in downloaded for f in categories['sample_videos']):
        print("2. 查看示例视频:")
        print(f"   ls -lh {args.output_dir}/v1-3/train_val/")
        
    if any(f in downloaded for f in categories['train_videos'] + categories['val_videos'] + categories['test_videos']):
        print("3. 解压视频文件:")
        print(f"   cd {args.output_dir}")
        print("   cat v1-2_train.tar.gz.* | tar xzf -")

if __name__ == '__main__':
    main()

