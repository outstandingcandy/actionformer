#!/usr/bin/env python3
"""
下载 ActionFormer 官方预训练模型

支持的预训练模型：
1. THUMOS14 + I3D
2. ActivityNet 1.3 + TSP
3. EPIC-Kitchens 100
4. Ego4D

官方链接：https://github.com/happyharrycn/actionformer_release
"""

import os
import argparse
import requests
from tqdm import tqdm


# 官方预训练模型链接
PRETRAINED_MODELS = {
    'thumos_i3d': {
        'name': 'THUMOS14 + I3D',
        'google_drive_id': '1isG3bc1dG5-llBRFCivJwz_7c_b0XDcY',
        'description': 'ActionFormer 在 THUMOS14 上训练（I3D 特征）',
        'performance': 'mAP@0.5: 70.95%',
        'checkpoint': 'thumos_i3d_reproduce/epoch_034.pth.tar',
        'config': 'configs/thumos_i3d.yaml'
    },
    'anet_tsp': {
        'name': 'ActivityNet 1.3 + TSP',
        'box_link': 'https://uwmadison.box.com/s/aisdoymowukc99zoc7gpqegxbb4whikx',
        'google_drive_id': '1VW8px1Nz9A17i0wMVUfxh6YsPCLVqL-S',
        'description': 'ActivityNet 1.3 特征和预训练模型（TSP 特征）',
        'performance': 'Average mAP: 36.56%',
        'file': 'anet_1.3.tar.gz',
        'md5': 'c415f50120b9425ee1ede9ac3ce11203'
    },
}


def download_from_google_drive(file_id, output_path):
    """
    从 Google Drive 下载文件
    
    Args:
        file_id: Google Drive 文件ID
        output_path: 输出路径
    """
    print(f"从 Google Drive 下载...")
    print(f"文件ID: {file_id}")
    
    # Google Drive 下载 URL
    url = f"https://drive.google.com/uc?id={file_id}&export=download"
    
    print("\n⚠️  注意：")
    print("Google Drive 大文件下载可能需要确认。")
    print("如果自动下载失败，请手动下载：")
    print(f"  1. 访问: https://drive.google.com/file/d/{file_id}/view")
    print(f"  2. 点击下载")
    print(f"  3. 保存到: {output_path}")
    print()
    
    try:
        # 尝试自动下载
        session = requests.Session()
        response = session.get(url, stream=True)
        
        # 检查是否需要确认
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                params = {'id': file_id, 'confirm': value, 'export': 'download'}
                response = session.get(url, params=params, stream=True)
                break
        
        # 获取文件大小
        total_size = int(response.headers.get('content-length', 0))
        
        # 下载文件
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        with open(output_path, 'wb') as f:
            if total_size == 0:
                f.write(response.content)
            else:
                progress_bar = tqdm(total=total_size, unit='B', unit_scale=True)
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        f.write(chunk)
                        progress_bar.update(len(chunk))
                progress_bar.close()
        
        print(f"✅ 下载完成: {output_path}")
        return True
        
    except Exception as e:
        print(f"❌ 自动下载失败: {e}")
        print("\n请手动下载文件。")
        return False


def show_available_models():
    """显示所有可用的预训练模型"""
    print("="*80)
    print("ActionFormer 官方预训练模型")
    print("="*80)
    
    for model_id, info in PRETRAINED_MODELS.items():
        print(f"\n{model_id}:")
        print(f"  名称: {info['name']}")
        print(f"  说明: {info['description']}")
        print(f"  性能: {info['performance']}")
        if 'google_drive_id' in info:
            print(f"  Google Drive ID: {info['google_drive_id']}")
        if 'box_link' in info:
            print(f"  Box 链接: {info['box_link']}")
    
    print("\n" + "="*80)


def download_model(model_id, output_dir='./pretrained'):
    """
    下载指定的预训练模型
    
    Args:
        model_id: 模型ID
        output_dir: 输出目录
    """
    if model_id not in PRETRAINED_MODELS:
        print(f"❌ 未知的模型ID: {model_id}")
        print(f"可用的模型: {list(PRETRAINED_MODELS.keys())}")
        return False
    
    info = PRETRAINED_MODELS[model_id]
    
    print("="*80)
    print(f"下载: {info['name']}")
    print("="*80)
    print(f"说明: {info['description']}")
    print(f"性能: {info['performance']}")
    print()
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 下载文件
    if 'google_drive_id' in info:
        if model_id == 'thumos_i3d':
            output_file = os.path.join(output_dir, 'thumos_i3d_reproduce.tar.gz')
        elif model_id == 'anet_tsp':
            output_file = os.path.join(output_dir, info.get('file', 'model.tar.gz'))
        else:
            output_file = os.path.join(output_dir, f'{model_id}.tar.gz')
        
        success = download_from_google_drive(info['google_drive_id'], output_file)
        
        if success:
            print(f"\n下载完成！")
            print(f"文件位置: {output_file}")
            
            # 解压说明
            print(f"\n后续步骤:")
            print(f"1. 解压文件:")
            print(f"   cd {output_dir}")
            print(f"   tar -xzf {os.path.basename(output_file)}")
            
            if 'config' in info:
                print(f"\n2. 使用预训练模型推理:")
                print(f"   python3 inference_activitynet.py \\")
                print(f"       --config {info['config']} \\")
                print(f"       --checkpoint {output_dir}/{info.get('checkpoint', 'model.pth.tar')} \\")
                print(f"       --annotation data/annotations.json \\")
                print(f"       --feature-dir data/features")
            
            return True
    
    else:
        print("⚠️  该模型需要手动下载")
        if 'box_link' in info:
            print(f"下载链接: {info['box_link']}")
        print(f"\n下载后，请将文件保存到: {output_dir}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='下载 ActionFormer 官方预训练模型',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

# 查看所有可用模型
python3 download_pretrained_models.py --list

# 下载 THUMOS14 预训练模型
python3 download_pretrained_models.py --model thumos_i3d

# 下载 ActivityNet 预训练模型
python3 download_pretrained_models.py --model anet_tsp --output-dir ./pretrained
        """
    )
    
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用的预训练模型')
    parser.add_argument('--model', type=str,
                        choices=list(PRETRAINED_MODELS.keys()),
                        help='要下载的模型ID')
    parser.add_argument('--output-dir', type=str, default='./pretrained',
                        help='输出目录（默认: ./pretrained）')
    
    args = parser.parse_args()
    
    if args.list:
        show_available_models()
    elif args.model:
        download_model(args.model, args.output_dir)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()

