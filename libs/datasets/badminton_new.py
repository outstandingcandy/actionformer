"""
Badminton dataset for new data format
"""

import os
import json
import torch
import numpy as np
from .datasets import register_dataset

@register_dataset("badminton_new")
class BadmintonNewDataset(torch.utils.data.Dataset):
    """
    新的羽毛球数据集类
    """
    
    def __init__(self, is_training, split, **kwargs):
        # 基础配置
        self.is_training = is_training
        self.split = split
        self.kwargs = kwargs
        
        # 数据路径
        self.json_file = kwargs.get('json_file', 'data/actionformer_badminton_new/badminton_annotations.json')
        self.feat_folder = kwargs.get('feat_folder', 'data/actionformer_badminton_new/features')
        self.file_ext = kwargs.get('file_ext', '.npy')
        self.split_file = kwargs.get('split_file', 'data/actionformer_badminton_new/splits.json')
        
        # 特征配置
        self.feat_stride = kwargs.get('feat_stride', 1)
        self.num_frames = kwargs.get('num_frames', 16)
        self.default_fps = kwargs.get('default_fps', 12.5)
        self.feat_offset = kwargs.get('feat_offset', 8.0)
        self.max_seq_len = kwargs.get('max_seq_len', 128)
        self.trunc_thresh = kwargs.get('trunc_thresh', 0.5)  # GT截断阈值
        
        # 数据配置
        self.input_dim = kwargs.get('input_dim', 2048)
        self.num_classes = kwargs.get('num_classes', 12)
        
        # 加载数据
        self._load_data()
        
        # 动作标签映射
        self._load_action_mapping()
        
        print(f"Loaded {len(self.data_list)} videos for {split} split")
    
    def _load_data(self):
        """加载数据"""
        # 加载标注文件
        with open(self.json_file, 'r', encoding='utf-8') as f:
            self.annotations = json.load(f)
        
        # 加载分割文件
        with open(self.split_file, 'r', encoding='utf-8') as f:
            splits = json.load(f)
        
        # 根据split选择视频
        if self.split[0] == 'train':
            video_ids = splits['train']
        elif self.split[0] == 'val':
            video_ids = splits['val']
        else:
            video_ids = list(self.annotations.keys())
        
        # 构建数据列表
        self.data_list = []
        for video_id in video_ids:
            if video_id in self.annotations:
                video_data = self.annotations[video_id]
                
                # 检查特征文件是否存在
                feat_file = os.path.join(self.feat_folder, f"{video_id}{self.file_ext}")
                if os.path.exists(feat_file):
                    self.data_list.append({
                        'id': video_id,
                        'duration': video_data['duration'],
                        'segments': video_data['segments'],
                        'labels': video_data['labels']
                    })
                else:
                    print(f"Warning: Feature file not found: {feat_file}")
    
    def _load_action_mapping(self):
        """加载动作映射"""
        mapping_file = os.path.join(os.path.dirname(self.json_file), 'action_mapping.json')
        if os.path.exists(mapping_file):
            with open(mapping_file, 'r', encoding='utf-8') as f:
                mapping_data = json.load(f)
                self.action_list = mapping_data['actions']
                self.action_to_id = mapping_data['action_to_id']
        else:
            # 如果没有映射文件，创建默认映射
            self.action_list = [f'action_{i}' for i in range(self.num_classes)]
            self.action_to_id = {action: i for i, action in enumerate(self.action_list)}
        
        # 创建标签字典
        self.label_dict = {i: action for action, i in self.action_to_id.items()}
    
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, index):
        """获取数据项"""
        video_item = self.data_list[index]
        
        # 加载特征
        feat_file = os.path.join(self.feat_folder, f"{video_item['id']}{self.file_ext}")
        feats = np.load(feat_file)  # shape: (T, C)
        
        # 转置为 (C, T) 格式
        feats = feats.T  # shape: (C, T)
        
        # # 截断或填充到最大长度
        # if feats.shape[1] > self.max_seq_len:
        #     feats = feats[:, :self.max_seq_len]
        # elif feats.shape[1] < self.max_seq_len:
        #     # 填充
        #     pad_width = self.max_seq_len - feats.shape[1]
        #     feats = np.pad(feats, ((0, 0), (0, pad_width)), mode='constant', constant_values=0)
        
        # 转换为tensor
        feats = torch.from_numpy(feats).float()
        
        # 处理动作标注
        if video_item['segments'] and video_item['labels']:
            # 转换时间到特征帧坐标
            segments = []
            for seg in video_item['segments']:
                start_time, end_time = seg
                # 转换为特征帧坐标
                start_frame = start_time * self.default_fps / self.feat_stride - self.feat_offset
                end_frame = end_time * self.default_fps / self.feat_stride - self.feat_offset
                segments.append([start_frame, end_frame])
            
            segments = torch.from_numpy(np.array(segments)).float()
            labels = torch.from_numpy(np.array(video_item['labels'])).long()
            
            # 过滤掉超出特征范围的GT标注（参考anet.py的处理）
            if self.is_training:
                vid_len = feats.shape[1]  # 实际特征长度
                valid_seg_list, valid_label_list = [], []
                for seg, label in zip(segments, labels):
                    if seg[0] >= vid_len:
                        # 跳过完全在特征范围外的动作
                        continue
                    # 计算动作在特征范围内的比例
                    ratio = (
                        (min(seg[1].item(), vid_len) - seg[0].item())
                        / (seg[1].item() - seg[0].item())
                    )
                    # 如果动作的大部分在范围内，保留并截断
                    if ratio >= self.trunc_thresh:
                        valid_seg_list.append(seg.clamp(max=vid_len))
                        valid_label_list.append(label.view(1))
                
                if len(valid_seg_list) > 0:
                    segments = torch.stack(valid_seg_list, dim=0)
                    labels = torch.cat(valid_label_list)
                else:
                    segments = torch.zeros((0, 2)).float()
                    labels = torch.zeros((0,)).long()
        else:
            segments = torch.zeros((0, 2)).float()
            labels = torch.zeros((0,)).long()
        
        # 返回数据
        return {
            'video_id': video_item['id'],
            'feats': feats,
            'segments': segments,
            'labels': labels,
            'fps': self.default_fps,
            'feat_stride': self.feat_stride,
            'feat_num_frames': self.num_frames,
            'feat_offset': self.feat_offset,
            'duration': video_item['duration']
        }
    
    def get_attributes(self):
        """获取数据集属性"""
        return {
            'dataset_name': 'badminton_new',
            'tiou_thresholds': np.linspace(0.1, 0.5, 5),  # 训练用较低阈值
            'empty_label_ids': [],
            'num_classes': self.num_classes,
            'action_list': self.action_list,
            'label_dict': self.label_dict
        }
