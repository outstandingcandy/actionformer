#!/usr/bin/env python3
"""
测试I3D特征提取功能
"""

import os
import sys
import torch
import numpy as np

# 添加路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from extract_features import SimpleFeatureExtractor

def test_i3d_init():
    """测试I3D初始化"""
    print("=" * 80)
    print("测试1: I3D特征提取器初始化")
    print("=" * 80)
    
    try:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"使用设备: {device}")
        
        extractor = SimpleFeatureExtractor(
            device=device,
            target_fps=25.0,
            feature_type='i3d',
            i3d_frame_count=32
        )
        
        print(f"✓ I3D特征提取器初始化成功")
        print(f"  - 特征维度: {extractor.feature_dim}")
        print(f"  - 窗口大小: {extractor.i3d_frame_count}帧")
        print(f"  - 目标FPS: {extractor.target_fps}")
        
        return extractor
        
    except Exception as e:
        print(f"✗ I3D初始化失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_i3d_extraction(extractor, video_path):
    """测试I3D特征提取"""
    print("\n" + "=" * 80)
    print("测试2: I3D特征提取")
    print("=" * 80)
    
    if not os.path.exists(video_path):
        print(f"✗ 测试视频不存在: {video_path}")
        print("请提供一个有效的视频路径进行测试")
        return None
    
    try:
        print(f"提取视频: {video_path}")
        
        features, fps = extractor.extract_video_features(video_path, target_fps=25.0)
        
        print(f"✓ 特征提取成功")
        print(f"  - 特征形状: {features.shape}")
        print(f"  - 特征维度: {features.shape[1]}")
        print(f"  - 时间步数: {features.shape[0]}")
        print(f"  - 原始FPS: {fps:.2f}")
        print(f"  - 特征数据类型: {features.dtype}")
        print(f"  - 特征值范围: [{features.min():.4f}, {features.max():.4f}]")
        
        return features
        
    except Exception as e:
        print(f"✗ 特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_i3d_config():
    """测试I3D配置文件"""
    print("\n" + "=" * 80)
    print("测试3: I3D配置文件检查")
    print("=" * 80)
    
    config_file = os.path.join(
        os.path.dirname(__file__), 
        'configs', 'recognition', 'i3d', 
        'i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py'
    )
    
    if os.path.exists(config_file):
        print(f"✓ I3D配置文件存在: {config_file}")
        return True
    else:
        print(f"✗ I3D配置文件不存在: {config_file}")
        return False

def main():
    print("\n" + "=" * 80)
    print("I3D特征提取功能测试")
    print("=" * 80 + "\n")
    
    # 测试1: 配置文件检查
    config_ok = test_i3d_config()
    
    if not config_ok:
        print("\n⚠️  配置文件缺失，无法继续测试")
        return
    
    # 测试2: 初始化
    extractor = test_i3d_init()
    
    if extractor is None:
        print("\n⚠️  初始化失败，无法继续测试")
        return
    
    # 测试3: 特征提取（如果有测试视频）
    # 检查常见的测试视频路径
    test_video_paths = [
        "/home/ubuntu/shuttle-sense/actionformer_release/0014_002.mp4",
        "/home/ubuntu/shuttle-sense/actionformer_release/0022_003.mp4",
        "/data/badminton/videos/0014_002.mp4",
    ]
    
    test_video = None
    for path in test_video_paths:
        if os.path.exists(path):
            test_video = path
            break
    
    if test_video:
        features = test_i3d_extraction(extractor, test_video)
        
        if features is not None:
            print("\n" + "=" * 80)
            print("✓ 所有测试通过！I3D特征提取功能正常")
            print("=" * 80)
        else:
            print("\n" + "=" * 80)
            print("⚠️  特征提取测试失败")
            print("=" * 80)
    else:
        print("\n" + "=" * 80)
        print("⚠️  未找到测试视频，跳过特征提取测试")
        print("提示: 您可以手动指定测试视频路径:")
        print("  python3 test_i3d_feature.py /path/to/video.mp4")
        print("=" * 80)

if __name__ == '__main__':
    if len(sys.argv) > 1:
        # 如果提供了视频路径，使用提供的路径
        test_video = sys.argv[1]
        extractor = test_i3d_init()
        if extractor:
            test_i3d_extraction(extractor, test_video)
    else:
        main()


