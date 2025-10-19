#!/usr/bin/env python3
"""
Feature extraction for ActionFormer - 支持多种特征提取方法
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
import cv2
import torchvision.models as models
import torchvision.transforms as transforms
from tqdm import tqdm
import argparse
import h5py

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Try to import I3D from mmaction2
try:
    # Use absolute path for mmaction2
    mmaction2_path = os.path.join(os.path.dirname(__file__), '..', '..', 'mmaction2')
    sys.path.insert(0, mmaction2_path)
    from mmaction.apis import init_recognizer
    from mmaction.utils import register_all_modules
    I3D_AVAILABLE = True
except ImportError:
    I3D_AVAILABLE = False
    print("Warning: mmaction2 not found. I3D features will not be available.")

# Try to import optical flow computation
try:
    FLOW_AVAILABLE = True
except ImportError:
    FLOW_AVAILABLE = False
    print("Warning: Optical flow computation will use basic OpenCV implementation.")


class ActionFormerFeatureExtractor:
    """ActionFormer专用特征提取器，支持多种特征类型."""
    
    def __init__(self, feature_type='resnet50', device='cuda', resolution='medium'):
        self.device = device
        self.feature_type = feature_type
        self.resolution = resolution
        
        # 设置视觉配置
        print(f"Using vision config: {self.vision_config}")
        
        if feature_type == 'resnet50':
            self._init_resnet50()
        elif feature_type == 'i3d':
            self._init_i3d()
        elif feature_type == 'two_stream':
            self._init_two_stream()
        elif feature_type == 'swin':
            self._init_swin()
        else:
            raise ValueError(f"Unsupported feature type: {feature_type}")
    
    def _init_resnet50(self):
        """初始化ResNet50特征提取器."""
        # 加载预训练ResNet50
        resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.feature_extractor = nn.Sequential(*list(resnet.children())[:-1])
        self.feature_extractor.to(self.device)
        self.feature_extractor.eval()
        
        self.feature_dim = 2048
        
        # 图像预处理
        image_size = self.vision_config.get_image_size()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.vision_config.mean, std=self.vision_config.std)
        ])
    
    def _init_two_stream(self):
        """初始化Two-Stream特征提取器."""
        print("Initializing Two-Stream (RGB + Flow) feature extractor...")
        
        # RGB stream - 使用ResNet50
        resnet_rgb = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        self.rgb_extractor = nn.Sequential(*list(resnet_rgb.children())[:-1])
        self.rgb_extractor.to(self.device)
        self.rgb_extractor.eval()
        
        # Flow stream - 使用ResNet50但修改输入通道
        resnet_flow = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # 修改第一层以接受2通道输入 (optical flow: dx, dy)
        original_conv1 = resnet_flow.conv1
        resnet_flow.conv1 = nn.Conv2d(
            2, 64, kernel_size=7, stride=2, padding=3, bias=False
        )
        
        # 初始化新的conv1层权重 - 取RGB权重的前2个通道的平均
        with torch.no_grad():
            rgb_weights = original_conv1.weight
            # 对于光流，我们使用RGB权重的平均来初始化
            flow_weights = rgb_weights[:, :2, :, :].clone()
            resnet_flow.conv1.weight = nn.Parameter(flow_weights)
        
        self.flow_extractor = nn.Sequential(*list(resnet_flow.children())[:-1])
        self.flow_extractor.to(self.device)
        self.flow_extractor.eval()
        
        # Two-Stream特征维度是两个stream的拼接
        self.feature_dim = 2048 + 2048  # RGB (2048) + Flow (2048) = 4096
        
        # 图像预处理
        image_size = self.vision_config.get_image_size()
        self.rgb_transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.vision_config.mean, std=self.vision_config.std)
        ])
        
        # 光流预处理 - 归一化到 [-1, 1]
        self.flow_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.0, 0.0], std=[1.0, 1.0])
        ])
    
    def _compute_optical_flow(self, frame1, frame2):
        """计算两帧之间的光流."""
        gray1 = cv2.cvtColor(frame1, cv2.COLOR_RGB2GRAY)
        gray2 = cv2.cvtColor(frame2, cv2.COLOR_RGB2GRAY)
        
        # 使用Farneback算法计算稠密光流
        flow = cv2.calcOpticalFlowFarneback(
            gray1, gray2, None, 0.5, 3, 15, 3, 5, 1.2, 0
        )
        
        # 光流范围限制和归一化
        flow = np.clip(flow, -20, 20)  # 限制光流幅度
        flow = flow / 20.0  # 归一化到[-1, 1]
        
        return flow
    
    def _init_i3d(self):
        """初始化I3D特征提取器."""
        if not I3D_AVAILABLE:
            raise ImportError("mmaction2 is required for I3D features")
        
        register_all_modules()
        
        # 加载I3D模型 - 使用绝对路径
        config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'mmaction2', 'configs', 'recognition', 'i3d', 'i3d_imagenet-pretrained-r50_8xb8-32x2x1-100e_kinetics400-rgb.py')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"I3D config file not found: {config_file}")
            
        self.model = init_recognizer(config_file, None, device=self.device)
        self.model.eval()
        
        self.clip_len = 32
        self.frame_interval = 2
        self.feature_dim = 2048
        
        # 图像预处理
        image_size = self.vision_config.get_image_size()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.vision_config.mean, std=self.vision_config.std)
        ])
    
    def _init_swin(self):
        """初始化Swin Transformer特征提取器."""
        if not I3D_AVAILABLE:
            raise ImportError("mmaction2 is required for Swin features")
        
        register_all_modules()
        
        # 加载Swin模型 - 使用绝对路径
        config_file = os.path.join(os.path.dirname(__file__), '..', '..', 'mmaction2', 'configs', 'recognition', 'swin', 'swin-tiny-p244-w877_in1k-pre_8xb8-amp-32x2x1-30e_kinetics400-rgb.py')
        if not os.path.exists(config_file):
            raise FileNotFoundError(f"Swin config file not found: {config_file}")
            
        self.model = init_recognizer(config_file, None, device=self.device)
        self.model.eval()
        
        self.feature_dim = 768  # Swin-Tiny feature dimension
        
        # 图像预处理
        image_size = self.vision_config.get_image_size()
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=self.vision_config.mean, std=self.vision_config.std)
        ])
    
    def extract_video_features(self, video_path, target_fps=25.0):
        """
        从视频提取特征.
        
        Args:
            video_path: 视频路径
            target_fps: 目标采样帧率
            
        Returns:
            features: numpy array, shape (T, feature_dim)
            fps: 原始视频帧率
        """
        if self.feature_type == 'resnet50':
            return self._extract_resnet_features(video_path, target_fps)
        elif self.feature_type == 'i3d':
            return self._extract_i3d_features(video_path, target_fps)
        elif self.feature_type == 'two_stream':
            return self._extract_two_stream_features(video_path, target_fps)
        elif self.feature_type == 'swin':
            return self._extract_swin_features(video_path, target_fps)
    
    def _extract_two_stream_features(self, video_path, target_fps=25.0):
        """使用Two-Stream方法提取RGB和光流特征."""
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
        
        # 读取所有需要的帧
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
        
        if len(all_frames) < 2:
            print("Warning: Not enough frames for optical flow computation")
            return np.zeros((1, self.feature_dim)), original_fps
        
        print(f"Processing {len(all_frames)} frames with Two-Stream...")
        
        features = []
        
        with torch.no_grad():
            for i in tqdm(range(len(all_frames)), desc="Two-Stream extraction"):
                # RGB特征
                rgb_frame = all_frames[i]
                rgb_tensor = self.rgb_transform(rgb_frame).unsqueeze(0).to(self.device)
                rgb_feature = self.rgb_extractor(rgb_tensor)
                rgb_feature = rgb_feature.squeeze().cpu().numpy()
                
                # 光流特征
                if i > 0:
                    # 计算当前帧和前一帧的光流
                    prev_frame = all_frames[i-1]
                    flow = self._compute_optical_flow(prev_frame, rgb_frame)
                    
                    # 转换光流为2通道tensor
                    flow_tensor = torch.from_numpy(flow.transpose(2, 0, 1)).float()  # HWC -> CHW
                    flow_tensor = transforms.Resize(self.vision_config.get_image_size())(flow_tensor)
                    flow_tensor = flow_tensor.unsqueeze(0).to(self.device)
                    
                    flow_feature = self.flow_extractor(flow_tensor)
                    flow_feature = flow_feature.squeeze().cpu().numpy()
                else:
                    # 第一帧没有光流，使用零向量
                    flow_feature = np.zeros(2048, dtype=np.float32)
                
                # 拼接RGB和Flow特征
                combined_feature = np.concatenate([rgb_feature, flow_feature])
                features.append(combined_feature)
        
        if features:
            features = np.array(features)
        else:
            features = np.zeros((1, self.feature_dim))
        
        print(f"Two-Stream features shape: {features.shape}")
        return features, original_fps
    
    def _extract_resnet_features(self, video_path, target_fps=25.0):
        """使用ResNet50提取帧级特征."""
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
        
        features = []
        frame_idx = 0
        
        with torch.no_grad():
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % frame_interval == 0:
                    # 转换BGR到RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    frame_tensor = self.transform(frame_rgb).unsqueeze(0).to(self.device)
                    
                    # 提取特征
                    feature = self.feature_extractor(frame_tensor)
                    feature = feature.squeeze().cpu().numpy()
                    features.append(feature)
                
                frame_idx += 1
        
        cap.release()
        
        if features:
            features = np.array(features)
        else:
            features = np.zeros((1, self.feature_dim))
        
        return features, original_fps
    
    def _extract_i3d_features(self, video_path, target_fps=25.0):
        """使用I3D提取时空特征."""
        # Note: target_fps parameter kept for API consistency but not used for I3D
        _ = target_fps  # Suppress unused variable warning
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")
        
        original_fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"Video: {total_frames} frames, {original_fps:.2f} fps")
        
        # 读取所有帧
        all_frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            all_frames.append(frame_rgb)
        
        cap.release()
        
        if len(all_frames) == 0:
            return np.zeros((1, self.feature_dim)), original_fps
        
        # 提取I3D特征clips
        features = []
        clip_stride = self.clip_len // 8 # 75% overlap
        
        with torch.no_grad():
            for start_idx in range(0, len(all_frames) - self.clip_len + 1, clip_stride):
                # 提取clip
                clip_frames = []
                for i in range(self.clip_len):
                    frame_idx = start_idx + i * self.frame_interval
                    if frame_idx < len(all_frames):
                        clip_frames.append(all_frames[frame_idx])
                    else:
                        clip_frames.append(all_frames[-1])
                
                # 预处理clip
                clip_tensors = []
                for frame in clip_frames:
                    frame_tensor = self.transform(frame)
                    clip_tensors.append(frame_tensor)
                
                # 转换为CTHW格式
                clip_tensor = torch.stack(clip_tensors, dim=1).unsqueeze(0).to(self.device)
                
                # 提取特征
                feature = self.model.backbone(clip_tensor)
                feature = torch.mean(feature, dim=[2, 3, 4])  # 全局平均池化
                feature = feature.squeeze().cpu().numpy()
                features.append(feature)
        
        if features:
            features = np.array(features)
        else:
            features = np.zeros((1, self.feature_dim))
        
        return features, original_fps
    
    def _extract_swin_features(self, video_path, target_fps=25.0):
        """使用Swin Transformer提取特征，处理完整的视频序列（内存优化版本）."""
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
        max_frames_per_batch = 32  # 限制每次处理的最大帧数
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


def convert_finebadminton_to_actionformer(data_dir, output_dir, feature_extractor):
    """
    将finebadminton数据集转换为ActionFormer格式.
    
    Args:
        data_dir: finebadminton数据集目录
        output_dir: 输出目录
        feature_extractor: 特征提取器
    """
    train_json = os.path.join(data_dir, 'train.json')
    if not os.path.exists(train_json):
        raise FileNotFoundError(f"Train file not found: {train_json}")
    
    # 解析数据
    with open(train_json, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    # 创建输出目录
    feat_dir = os.path.join(output_dir, 'features')
    os.makedirs(feat_dir, exist_ok=True)
    
    # 处理数据
    actionformer_db = {}
    action_types = set()
    
    print(f"Processing {len(data)} videos...")
    
    for item in tqdm(data, desc="Converting videos"):
        video_id = item['id']
        video_path = os.path.join(data_dir, item['video'])
        
        if not os.path.exists(video_path):
            print(f"Warning: Video not found: {video_path}")
            continue
        
        try:
            # 提取特征
            features, fps = feature_extractor.extract_video_features(video_path)
            
            # 保存特征到npy文件
            feat_file = os.path.join(feat_dir, f"{video_id}.npy")
            np.save(feat_file, features)
            
            # 解析标注
            segments = []
            labels = []
            total_frames = 0
            
            for conv in item['conversations']:
                if conv['from'] == 'assistant':
                    try:
                        response_data = json.loads(conv['value'])
                        annotations = response_data.get('detailed_annotations', [])
                        
                        for ann in annotations:
                            temporal_info = ann.get('temporal_info', {})
                            start_frame = temporal_info.get('start_frame', 0)
                            end_frame = temporal_info.get('end_frame', 0)
                            hit_type = temporal_info.get('hit_type', 'unknown')
                            
                            if start_frame < end_frame:
                                # 转换帧数为时间
                                start_time = start_frame / fps
                                end_time = end_frame / fps
                                total_frames = end_frame
                                segments.append([start_time, end_time])
                                labels.append(hit_type)
                                action_types.add(hit_type)

                        break
                    except json.JSONDecodeError:
                        print(f"Warning: Could not parse annotations for {video_id}")
            
            # 计算视频时长 - 使用原始视频信息而不是特征数量
            duration = total_frames / fps
            
            actionformer_db[video_id] = {
                'duration': duration,
                'segments': segments,
                'labels': labels
            }
            
        except Exception as e:
            print(f"Error processing {video_id}: {e}")
            continue
    
    # 创建动作映射
    action_list = sorted(list(action_types))
    action_to_id = {action: idx for idx, action in enumerate(action_list)}
    
    # 转换标签为ID
    for vid_data in actionformer_db.values():
        vid_data['labels'] = [action_to_id[label] for label in vid_data['labels']]
    
    # 保存ActionFormer格式的数据
    output_json = os.path.join(output_dir, 'actionformer_annotations.json')
    with open(output_json, 'w') as f:
        json.dump(actionformer_db, f, indent=2)
    
    # 保存动作映射
    mapping_file = os.path.join(output_dir, 'action_mapping.json')
    with open(mapping_file, 'w') as f:
        json.dump({
            'actions': action_list,
            'action_to_id': action_to_id
        }, f, indent=2, ensure_ascii=False)
    
    print(f"\\nConversion completed!")
    print(f"Processed {len(actionformer_db)} videos")
    print(f"Features saved to: {feat_dir}")
    print(f"Annotations saved to: {output_json}")
    print(f"Action mapping saved to: {mapping_file}")
    print(f"Action types: {action_list}")
    
    return actionformer_db, action_to_id


def save_features_to_hdf5(feat_dir, hdf5_path):
    """将numpy特征文件转换为HDF5格式."""
    print(f"Converting features to HDF5: {hdf5_path}")
    
    with h5py.File(hdf5_path, 'w') as hf:
        feat_files = [f for f in os.listdir(feat_dir) if f.endswith('.npy')]
        
        for feat_file in tqdm(feat_files, desc="Converting to HDF5"):
            video_id = os.path.splitext(feat_file)[0]
            feat_path = os.path.join(feat_dir, feat_file)
            features = np.load(feat_path)
            hf.create_dataset(video_id, data=features)
    
    print(f"HDF5 file saved: {hdf5_path}")


def main():
    parser = argparse.ArgumentParser(description='Extract features for ActionFormer')
    parser.add_argument('--data_dir', type=str, default='../../data/finebadminton_qwen_dataset',
                        help='Path to finebadminton dataset')
    parser.add_argument('--output_dir', type=str, default='../../data/actionformer_badminton',
                        help='Output directory')
    parser.add_argument('--feature_type', type=str, default='resnet50', 
                        choices=['resnet50', 'i3d', 'two_stream', 'swin'],
                        help='Feature extraction method')
    parser.add_argument('--resolution', type=str, default='medium',
                        choices=['low', 'medium', 'high', 'ultra'],
                        help='Video processing resolution')
    parser.add_argument('--target_fps', type=float, default=25.0,
                        help='Target sampling frame rate')
    parser.add_argument('--save_hdf5', action='store_true',
                        help='Also save features in HDF5 format')
    
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # 创建特征提取器
    feature_extractor = ActionFormerFeatureExtractor(
        feature_type=args.feature_type,
        device=device,
        resolution=args.resolution
    )
    
    # 转换数据
    os.makedirs(args.output_dir, exist_ok=True)
    
    convert_finebadminton_to_actionformer(
        args.data_dir, args.output_dir, feature_extractor
    )
    
    # 可选：转换为HDF5格式
    if args.save_hdf5:
        feat_dir = os.path.join(args.output_dir, 'features')
        hdf5_path = os.path.join(args.output_dir, f'features_{args.feature_type}.hdf5')
        save_features_to_hdf5(feat_dir, hdf5_path)


if __name__ == '__main__':
    main()