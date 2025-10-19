#!/usr/bin/env python3
"""
为新的羽毛球数据提取特征
"""

import os
import sys
import json
import numpy as np
import torch
import cv2
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import traceback

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import video_swin from timm
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False
    print("Warning: timm not found. Swin features will not be available.")

# mmaction2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'mmaction2')
# sys.path.insert(0, mmaction2_path)
from mmaction.apis import init_recognizer
from mmaction.utils import register_all_modules
from mmcv.transforms import Compose
from mmengine.dataset import default_collate

class SimpleFeatureExtractor:
    """特征提取器，支持ResNet50、Swin、SwinV2 (256/384)、Uniformer和I3D"""
    
    def __init__(self, device='cuda', target_fps=12.5, feature_type='resnet50', 
                 uniformer_frame_count=8, window_stride=None, i3d_frame_count=32):
        self.device = device
        self.target_fps = target_fps
        self.feature_type = feature_type
        self.uniformer_frame_count = uniformer_frame_count
        self.i3d_frame_count = i3d_frame_count  # I3D默认使用32帧
        # 如果没有指定 window_stride，默认为 uniformer_frame_count 的一半
        self.window_stride = window_stride if window_stride is not None else uniformer_frame_count // 2
        
        if feature_type == 'video_swin':
            self._init_swin()
        elif feature_type == 'uniformer':
            self._init_uniformer()
        elif feature_type == 'i3d':
            self._init_i3d()
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    
    def _init_swin(self):
        """初始化Swin Transformer特征提取器."""
        register_all_modules()
        
        # 加载Swin模型 - 使用绝对路径
        config_file = os.path.join(os.path.dirname(__file__), 'configs', 'recognition', 'swin', 'swin-large-p244-w877_in22k-pre_16xb8-amp-32x2x1-30e_kinetics700-rgb.py')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Swin config file not found: {config_file}")
            
        self.model = init_recognizer(config_file, None, device=self.device)
        self.model.eval()

        self.pipeline = Compose(self.model.cfg.test_pipeline)  # Not needed for feature extraction
        
        self.feature_dim = 768  # Swin-Tiny feature dimension
        
        # 图像预处理
        image_size = (224, 224)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),  # 保持长宽比，将较短边resize到256
            transforms.CenterCrop(image_size),  # center crop到224x224
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def _init_uniformer(self):
        """初始化Uniformer特征提取器."""
        register_all_modules()
        
        # 加载Uniformer模型 - 使用绝对路径
        config_file = os.path.join(os.path.dirname(__file__), 'configs', 'recognition', 'uniformerv2', 'uniformerv2-large-p14-res336_clip-kinetics710-pre_u32_kinetics700-rgb.py')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Uniformer config file not found: {config_file}")
            
        self.model = init_recognizer(config_file, None, device=self.device)
        self.model.eval()

        self.pipeline = Compose(self.model.cfg.test_pipeline)
        
        self.feature_dim = 1024  # Uniformer-Large feature dimension
        
        # 图像预处理 - Uniformer使用336x336分辨率
        image_size = (336, 336)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(336),  # 保持长宽比，将较短边resize到384
            transforms.CenterCrop(image_size),  # center crop到336x336
            transforms.ToTensor(),
            transforms.Normalize(mean=[114.75, 114.75, 114.75], std=[57.375, 57.375, 57.375])
        ])

    def _init_i3d(self):
        """初始化I3D特征提取器."""
        register_all_modules()
        
        # 加载I3D模型 - 使用绝对路径
        config_file = os.path.join(os.path.dirname(__file__), 'configs', 'recognition', 'i3d', 'i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"I3D config file not found: {config_file}")
            
        self.model = init_recognizer(config_file, None, device=self.device)
        self.model.eval()

        self.pipeline = Compose(self.model.cfg.test_pipeline)
        
        self.feature_dim = 2048  # I3D-ResNet50 feature dimension
        
        # 图像预处理 - I3D使用224x224分辨率
        image_size = (224, 224)
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(256),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])

    def extract_feature(self, video_path, device='cuda:0'):
        """提取单个视频的特征（使用pipeline方式）"""
        # 获取视频的帧率
        cap = cv2.VideoCapture(video_path)
        original_fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 25.0
        cap.release()
        
        data = dict(filename=video_path, label=-1, start_index=0, modality='RGB')
        data = self.pipeline(data)
        data = default_collate([data])  # 替换 collate

        # 数据送到 GPU
        for k, v in data.items():
            if isinstance(v, torch.Tensor):
                data[k] = v.to(device).float()

        with torch.no_grad():
            feature = self.model.extract_feat(data['inputs'])  # [B, C, T', H', W']
            feat_spatial = feature.mean(dim=[3, 4])     # [B, C, T']
            feat_seq = feat_spatial.permute(0, 2, 1)    # [B, T', C]
        return feat_seq.squeeze(0).cpu(), original_fps
    
    def extract_video_features(self, video_path, target_fps=25.0):
        """提取特征，根据特征类型选择相应的提取方法"""
        if self.feature_type == 'video_swin':
            return self.extract_video_features_swin(video_path, target_fps)
        elif self.feature_type == 'uniformer':
            return self.extract_video_features_uniformer(video_path, target_fps)
        elif self.feature_type == 'i3d':
            return self.extract_video_features_i3d(video_path, target_fps)
        else:
            raise ValueError(f"Unsupported feature type: {self.feature_type}")

    def extract_video_features_swin(self, video_path, target_fps=25.0):
        """使用Swin Transformer提取特征，处理完整的视频序列（内存优化版本） - 参考extract_features.py"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {total_frames} frames, {original_fps:.2f} fps")
        
        # 计算采样间隔
        if target_fps is not None and original_fps > target_fps:
            frame_interval = int(original_fps / target_fps)
        else:
            frame_interval = 1
        
        # 读取并采样帧
        all_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        
        if len(all_frames) == 0:
            return np.zeros((1, self.feature_dim)), original_fps
        
        print(f"Processing {len(all_frames)} frames with Swin (memory optimized)...")
        
        # 内存优化：使用滑动窗口处理长视频
        max_frames_per_batch = 64  # 限制每次处理的最大帧数
        overlap = 8  # 窗口重叠帧数，保持时序连续性
        swin_downsample_ratio = 2  # Swin的时序下采样率是2:1
        
        all_features = []
        
        # 预处理所有帧
        print("Preprocessing frames...")
        frame_tensors = []
        for frame in tqdm(all_frames, desc="Preprocessing"):
            frame_tensor = self.transform(frame)
            frame_tensors.append(frame_tensor)
        
        # 分段处理长视频
        start_idx = 0
        while start_idx < len(frame_tensors):
            end_idx = min(start_idx + max_frames_per_batch, len(frame_tensors))
            
            # 提取当前窗口的帧
            window_tensors = frame_tensors[start_idx:end_idx]
            window_size = len(window_tensors)
            
            print(f"Processing frames {start_idx}-{end_idx-1} ({window_size} frames)")
            
            # 构建窗口的视频序列张量
            video_tensor = torch.stack(window_tensors, dim=1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                try:
                    # 提取特征
                    features = self.model.backbone(video_tensor)
                    
                    # 处理输出维度
                    if len(features.shape) == 5:  # (B, C, T, H, W)
                        features = torch.mean(features, dim=[3, 4])  # 空间平均池化: (B, C, T)
                        features = features.permute(0, 2, 1)  # (B, C, T) -> (B, T, C)
                    elif len(features.shape) == 4:  # (B, C, H, W)
                        features = torch.mean(features, dim=[2, 3])  # 空间平均池化: (B, C)
                        features = features.unsqueeze(1)  # 添加时间维度: (B, 1, C)
                    
                    window_features = features.squeeze(0).cpu().numpy()  # (T, C)
                    
                    # 处理重叠部分 - 考虑Swin的2:1下采样
                    # 输入帧的重叠对应输出特征的重叠
                    overlap_features = overlap // swin_downsample_ratio
                    
                    if start_idx > 0:
                        # 跳过重叠的前半部分特征
                        window_features = window_features[overlap_features//2:]
                    if end_idx < len(frame_tensors):
                        # 跳过重叠的后半部分特征
                        window_features = window_features[:-overlap_features//2]
                    
                    all_features.append(window_features)
                    
                except torch.cuda.OutOfMemoryError:
                    print(f"CUDA OOM at window {start_idx}-{end_idx}, reducing batch size...")
                    # 减小窗口大小重试
                    max_frames_per_batch = max(16, max_frames_per_batch // 2)
                    continue
            
            # 清理GPU缓存
            torch.cuda.empty_cache()
            
            # 移动窗口
            if end_idx >= len(frame_tensors):
                break
            start_idx = end_idx - overlap
        
        # 合并所有特征
        if all_features:
            features = np.concatenate(all_features, axis=0)
        else:
            features = np.zeros((1, self.feature_dim))
        
        print(f"Swin features shape: {features.shape}")
        return features, original_fps

    def extract_video_features_uniformer(self, video_path, target_fps=25.0):
        """使用Uniformer提取特征，处理完整的视频序列（内存优化版本）"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {total_frames} frames, {original_fps:.2f} fps")
        
        # 计算采样间隔
        if target_fps is not None and original_fps > target_fps:
            frame_interval = int(original_fps / target_fps)
        else:
            frame_interval = 1
        
        # 读取并采样帧
        all_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        
        if len(all_frames) == 0:
            return np.zeros((1, self.feature_dim)), original_fps
        
        print(f"Processing {len(all_frames)} frames with Uniformer (memory optimized)...")

        # 使用实例变量中的参数
        uniformer_frame_count = self.uniformer_frame_count
        window_stride = self.window_stride

        # 预处理所有帧
        print("Preprocessing frames...")
        frame_tensors = []
        for frame in tqdm(all_frames, desc="Preprocessing"):
            frame_tensor = self.transform(frame)
            frame_tensors.append(frame_tensor)
        
        print(f"Processing {len(frame_tensors)} frames with sliding window (stride={window_stride}, overlap={uniformer_frame_count - window_stride})")
        
        window_features_list = []
        window_idx = 0
        start_idx = 0
        while start_idx < len(frame_tensors):
            end_idx = min(start_idx + uniformer_frame_count, len(frame_tensors))
            window_idx += 1
            
            # 提取当前窗口的帧
            window_tensors = frame_tensors[start_idx:end_idx]
            window_size = len(window_tensors)
            
            # 如果窗口大小小于uniformer_frame_count，用最后一帧填充
            if window_size < uniformer_frame_count:
                last_frame = window_tensors[-1]
                while len(window_tensors) < uniformer_frame_count:
                    window_tensors.append(last_frame)
                print(f"Window {window_idx}: Padding with {uniformer_frame_count - window_size} repeated frames")
            
            print(f"Window {window_idx}: Processing frames {start_idx}-{end_idx-1} ({len(window_tensors)} frames)")
            
            # 构建视频序列张量 (B, C, T, H, W)
            video_tensor = torch.stack(window_tensors, dim=1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # 使用正确的特征提取方法
                data = {'inputs': video_tensor}
                # add dim for uniformer
                data['inputs'] = data['inputs'].unsqueeze(1)
                features = self.model.extract_feat(data['inputs'])[0]
                
                window_features = features.cpu().numpy()  # (T, C)
                window_features_list.append(window_features)
                
                print(f"  Window {window_idx} completed: {window_features.shape[0]} features added")
                
                # 清理GPU缓存
                torch.cuda.empty_cache()
                
                # 移动到下一个窗口（滑动）
                start_idx += window_stride
                if start_idx >= len(frame_tensors):
                    break
        
        features = np.concatenate(window_features_list, axis=0)
        print(f"Uniformer features shape: {features.shape}")
        return features, original_fps

    def extract_video_features_i3d(self, video_path, target_fps=25.0):
        """使用I3D提取特征，处理完整的视频序列（滑动窗口版本）"""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {total_frames} frames, {original_fps:.2f} fps")
        
        # 计算采样间隔
        if target_fps is not None and original_fps > target_fps:
            frame_interval = int(original_fps / target_fps)
        else:
            frame_interval = 1
        
        # 读取并采样帧
        all_frames = []
        frame_idx = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            if frame_idx % frame_interval == 0:
                # 转换BGR到RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                all_frames.append(frame_rgb)
            
            frame_idx += 1
        
        cap.release()
        
        if len(all_frames) == 0:
            return np.zeros((1, self.feature_dim)), original_fps
        
        print(f"Processing {len(all_frames)} frames with I3D (sliding window)...")

        # 使用滑动窗口处理
        i3d_frame_count = self.i3d_frame_count
        window_stride = i3d_frame_count // 2  # I3D默认步长为窗口的一半

        # 预处理所有帧
        print("Preprocessing frames...")
        frame_tensors = []
        for frame in tqdm(all_frames, desc="Preprocessing"):
            frame_tensor = self.transform(frame)
            frame_tensors.append(frame_tensor)
        
        print(f"Processing {len(frame_tensors)} frames with sliding window (window_size={i3d_frame_count}, stride={window_stride})")
        
        window_features_list = []
        window_idx = 0
        start_idx = 0
        
        while start_idx < len(frame_tensors):
            end_idx = min(start_idx + i3d_frame_count, len(frame_tensors))
            window_idx += 1
            
            # 提取当前窗口的帧
            window_tensors = frame_tensors[start_idx:end_idx]
            window_size = len(window_tensors)
            
            # 如果窗口大小小于i3d_frame_count，用最后一帧填充
            if window_size < i3d_frame_count:
                last_frame = window_tensors[-1]
                while len(window_tensors) < i3d_frame_count:
                    window_tensors.append(last_frame)
                print(f"Window {window_idx}: Padding with {i3d_frame_count - window_size} repeated frames")
            
            print(f"Window {window_idx}: Processing frames {start_idx}-{end_idx-1} ({len(window_tensors)} frames)")
            
            # 构建视频序列张量 (B, C, T, H, W)
            video_tensor = torch.stack(window_tensors, dim=1).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                # 使用I3D提取特征
                data = {'inputs': video_tensor}
                features = self.model.extract_feat(data['inputs'])  # [B, C, T', H', W']
                
                # 空间平均池化
                if len(features.shape) == 5:  # (B, C, T, H, W)
                    features = torch.mean(features, dim=[3, 4])  # (B, C, T)
                    features = features.permute(0, 2, 1)  # (B, T, C)
                elif len(features.shape) == 3:  # (B, T, C)
                    pass  # 已经是正确的形状
                else:  # (B, C)
                    features = features.unsqueeze(1)  # (B, 1, C)
                
                window_features = features.squeeze(0).cpu().numpy()  # (T, C)
                window_features_list.append(window_features)
                
                print(f"  Window {window_idx} completed: {window_features.shape[0]} features added")
                
                # 清理GPU缓存
                torch.cuda.empty_cache()
                
                # 移动到下一个窗口（滑动）
                start_idx += window_stride
                if start_idx >= len(frame_tensors):
                    break
        
        features = np.concatenate(window_features_list, axis=0)
        print(f"I3D features shape: {features.shape}")
        return features, original_fps

def extract_features_for_badminton(annotations_file, video_dir, output_dir, feature_extractor):
    """
    为羽毛球数据提取特征
    
    Args:
        annotations_file: 标注文件路径
        video_dir: 视频目录
        output_dir: 输出目录
        feature_extractor: 特征提取器
    """
    
    # 创建输出目录
    feat_dir = os.path.join(output_dir, 'features')
    os.makedirs(feat_dir, exist_ok=True)
    
    # 读取标注文件
    with open(annotations_file, 'r', encoding='utf-8') as f:
        annotations = json.load(f)
    
    print(f"处理 {len(annotations)} 个视频...")
    
    # 提取特征
    success_count = 0
    for video_id, _ in tqdm(annotations.items(), desc="提取特征"):
        video_file = f"{video_id}.mp4"
        video_path = os.path.join(video_dir, video_file)
        
        if not os.path.exists(video_path):
            print(f"警告: 视频文件不存在: {video_path}")
            continue
        
        try:
            # 提取特征
            features, _ = feature_extractor.extract_video_features(video_path, target_fps=feature_extractor.target_fps)
            # 保存特征到npy文件
            feat_file = os.path.join(feat_dir, f"{video_id}.npy")
            np.save(feat_file, features)
            
            success_count += 1
            
        except Exception as e:
            print(f"错误: 处理视频 {video_id} 时出错: {e}")
            print(traceback.format_exc())
            continue
    
    print(f"\n特征提取完成!")
    print(f"成功处理: {success_count}/{len(annotations)} 个视频")
    print(f"特征保存到: {feat_dir}")

def extract_features_for_activitynet(annotations_file, video_dir, output_dir, feature_extractor, 
                                     subset=None, file_prefix='v_', video_ext='.mp4'):
    """
    为 ActivityNet 数据提取特征
    
    ActivityNet 标注格式：
    {
      "database": {
        "video_id": {
          "subset": "training",
          "duration": 120.5,
          "annotations": [
            {
              "label": "Action Name",
              "segment": [10.2, 55.8]
            }
          ]
        }
      }
    }
    
    Args:
        annotations_file: ActivityNet 标注文件路径
        video_dir: 视频目录
        output_dir: 输出目录
        feature_extractor: 特征提取器
        subset: 要处理的子集 ('training', 'validation', 'testing', None=全部)
        file_prefix: 视频文件前缀（默认 'v_'）
        video_ext: 视频文件扩展名（默认 '.mp4'）
    """
    
    # 创建输出目录
    feat_dir = os.path.join(output_dir, 'features')
    os.makedirs(feat_dir, exist_ok=True)
    
    # 读取标注文件
    print(f"读取标注文件: {annotations_file}")
    with open(annotations_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 获取 database
    if 'database' in data:
        database = data['database']
        print(f"✓ ActivityNet 格式")
    else:
        # 可能是其他格式
        database = data
        print(f"✓ 简单字典格式")
    
    # 统计信息
    subset_stats = {}
    for video_id, video_info in database.items():
        video_subset = video_info.get('subset', 'unknown')
        subset_stats[video_subset] = subset_stats.get(video_subset, 0) + 1
    
    print(f"\n数据集统计:")
    print(f"  总视频数: {len(database)}")
    for sub, count in sorted(subset_stats.items()):
        print(f"  {sub}: {count}")
    
    # 筛选要处理的视频
    if subset:
        videos_to_process = {
            vid: info for vid, info in database.items() 
            if info.get('subset', '').lower() == subset.lower()
        }
        print(f"\n筛选 '{subset}' 子集: {len(videos_to_process)} 个视频")
    else:
        videos_to_process = database
        print(f"\n处理所有视频: {len(videos_to_process)} 个")
    
    # 提取特征
    success_count = 0
    failed_videos = []
    
    print(f"\n开始提取特征...")
    print(f"视频目录: {video_dir}")
    print(f"文件前缀: {file_prefix}")
    print(f"文件扩展名: {video_ext}")
    print("="*80)
    
    for video_id, video_info in tqdm(videos_to_process.items(), desc="提取特征"):
        # 构建视频文件路径
        video_file = f"{file_prefix}{video_id}{video_ext}"
        video_path = os.path.join(video_dir, video_file)
        
        # 检查文件是否存在
        if not os.path.exists(video_path):
            # 尝试不带前缀
            video_file_alt = f"{video_id}{video_ext}"
            video_path_alt = os.path.join(video_dir, video_file_alt)
            if os.path.exists(video_path_alt):
                video_path = video_path_alt
            else:
                print(f"\n警告: 视频文件不存在: {video_path}")
                failed_videos.append(video_id)
                continue
        
        try:
            # 提取特征
            features, fps = feature_extractor.extract_video_features(video_path)
            
            # 保存特征到 npy 文件
            feat_file = os.path.join(feat_dir, f"{file_prefix}{video_id}.npy")
            np.save(feat_file, features)
            
            success_count += 1
            
            # 每处理100个视频显示一次进度
            if success_count % 100 == 0:
                print(f"\n已处理: {success_count}/{len(videos_to_process)} 个视频")
            
        except Exception as e:
            print(f"\n错误: 处理视频 {video_id} 时出错: {e}")
            print(f"  视频路径: {video_path}")
            failed_videos.append(video_id)
            continue
    
    # 打印摘要
    print("\n" + "="*80)
    print("特征提取完成!")
    print("="*80)
    print(f"成功处理: {success_count}/{len(videos_to_process)} 个视频")
    print(f"失败: {len(failed_videos)} 个视频")
    print(f"成功率: {success_count/len(videos_to_process)*100:.1f}%")
    print(f"\n特征保存到: {feat_dir}")
    
    # 保存失败列表
    if failed_videos:
        failed_file = os.path.join(output_dir, 'failed_videos.txt')
        with open(failed_file, 'w') as f:
            for vid in failed_videos:
                f.write(f"{vid}\n")
        print(f"失败视频列表已保存: {failed_file}")
        print(f"\n失败的视频 (前20个):")
        for vid in failed_videos[:20]:
            print(f"  - {vid}")
        if len(failed_videos) > 20:
            print(f"  ... 还有 {len(failed_videos)-20} 个")

def main():
    parser = argparse.ArgumentParser(
        description='视频特征提取工具',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
使用示例:

1. 提取 Badminton 数据特征:
   python3 extract_features.py \\
       --annotations_file data/badminton/annotations.json \\
       --video_dir data/badminton/videos \\
       --output_dir data/badminton_features \\
       --feature_type uniformer \\
       --dataset_format badminton

2. 提取 ActivityNet 数据特征（全部）:
   python3 extract_features.py \\
       --annotations_file data/activitynet_hf/activity_net.v1-3.min.json \\
       --video_dir data/activitynet_videos \\
       --output_dir data/activitynet_features \\
       --feature_type uniformer \\
       --dataset_format activitynet

3. 提取 ActivityNet 验证集特征:
   python3 extract_features.py \\
       --annotations_file data/activitynet_hf/activity_net.v1-3.min.json \\
       --video_dir data/activitynet_videos \\
       --output_dir data/activitynet_features \\
       --feature_type uniformer \\
       --dataset_format activitynet \\
       --subset validation
        """
    )
    
    parser.add_argument('--annotations_file', type=str, 
                        default='data/actionformer_badminton_new/badminton_annotations.json',
                        help='标注文件路径')
    parser.add_argument('--video_dir', type=str, 
                        default='data/videos',
                        help='视频文件目录')
    parser.add_argument('--output_dir', type=str, 
                        default='data/actionformer_badminton_new',
                        help='输出目录')
    parser.add_argument('--target_fps', type=float, default=25.0,
                        help='目标采样帧率（默认: 25.0）')
    parser.add_argument('--feature_type', type=str, default='video_swin', 
                        choices=['resnet50', 'swin', 'swinv2', 'swinv2_384', 'video_swin', 'uniformer', 'i3d'],
                        help='特征类型: resnet50 (2048维), swin (768维, 224×224), swinv2 (768维, 256×256), swinv2_384 (1024维, 384×384), uniformer (1024维, 336×336), i3d (2048维, 224×224)')
    parser.add_argument('--dataset_format', type=str, default='badminton',
                        choices=['badminton', 'activitynet'],
                        help='数据集格式: badminton（简单字典）或 activitynet（标准格式）')
    parser.add_argument('--subset', type=str, default=None,
                        choices=['training', 'validation', 'testing'],
                        help='ActivityNet 子集（仅用于 activitynet 格式）')
    parser.add_argument('--file_prefix', type=str, default='v_',
                        help='视频文件前缀（默认: v_，用于 ActivityNet）')
    parser.add_argument('--video_ext', type=str, default='.mp4',
                        help='视频文件扩展名（默认: .mp4）')
    parser.add_argument('--uniformer_frame_count', type=int, default=8,
                        help='Uniformer 每个窗口的帧数（默认: 8）')
    parser.add_argument('--window_stride', type=int, default=None,
                        help='滑动窗口的步长（默认: uniformer_frame_count//2）')
    parser.add_argument('--i3d_frame_count', type=int, default=32,
                        help='I3D 每个窗口的帧数（默认: 32）')
    
    args = parser.parse_args()
    
    # 检查输入文件
    if not os.path.exists(args.annotations_file):
        print(f"错误: 标注文件不存在: {args.annotations_file}")
        return
    
    if not os.path.exists(args.video_dir):
        print(f"错误: 视频目录不存在: {args.video_dir}")
        return
    
    # 设置设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("="*80)
    print("特征提取配置")
    print("="*80)
    print(f"设备: {device}")
    print(f"特征类型: {args.feature_type}")
    print(f"数据集格式: {args.dataset_format}")
    print(f"目标 FPS: {args.target_fps}")
    if args.dataset_format == 'activitynet':
        print(f"子集: {args.subset if args.subset else '全部'}")
        print(f"文件前缀: {args.file_prefix}")
    print("="*80)
    
    # 根据特征类型显示提示
    feature_dim_info = {
        'resnet50': 2048,
        'swin': 768,
        'swinv2': 768,
        'swinv2_384': 1024,
        'video_swin': 768,
        'uniformer': 1024,
        'i3d': 2048
    }
    feat_dim = feature_dim_info.get(args.feature_type, 'unknown')
    print(f"\n⚠️  特征维度: {feat_dim}")
    if feat_dim != 2048:
        print(f"   训练时请确保配置文件中 input_dim={feat_dim}")
    print()
    
    # 创建特征提取器
    feature_extractor = SimpleFeatureExtractor(
        device=device,
        target_fps=args.target_fps,
        feature_type=args.feature_type,
        uniformer_frame_count=args.uniformer_frame_count,
        window_stride=args.window_stride,
        i3d_frame_count=args.i3d_frame_count
    )
    
    # 根据数据集格式调用不同的提取函数
    if args.dataset_format == 'activitynet':
        extract_features_for_activitynet(
            annotations_file=args.annotations_file,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            feature_extractor=feature_extractor,
            subset=args.subset,
            file_prefix=args.file_prefix,
            video_ext=args.video_ext
        )
    elif args.dataset_format == 'badminton':
        # 保持向后兼容：自动更新输出目录名称
        if args.output_dir == 'data/actionformer_badminton_new':
            if args.feature_type == 'swin':
                args.output_dir = 'data/actionformer_badminton_swin'
            elif args.feature_type == 'swinv2':
                args.output_dir = 'data/actionformer_badminton_swinv2'
            elif args.feature_type == 'swinv2_384':
                args.output_dir = 'data/actionformer_badminton_swinv2_384'
            elif args.feature_type == 'uniformer':
                args.output_dir = 'data/actionformer_badminton_uniformer'
            print(f"输出目录已更新为: {args.output_dir}")
        
        extract_features_for_badminton(
            annotations_file=args.annotations_file,
            video_dir=args.video_dir,
            output_dir=args.output_dir,
            feature_extractor=feature_extractor
        )
    else:
        print(f"错误: 未知的数据集格式: {args.dataset_format}")
        return

if __name__ == '__main__':
    main()
