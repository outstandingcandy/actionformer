#!/usr/bin/env python3
"""
ActivityNet数据集读取和评估示例
展示如何使用ActivityNet数据集进行训练和评估
"""

import os
import sys
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from libs.datasets import make_dataset, make_data_loader
from libs.utils import ANETdetection
from libs.core import load_config


class ActivityNetDatasetExample:
    """ActivityNet数据集使用示例类"""
    
    def __init__(self, config_file=None):
        """
        初始化ActivityNet数据集
        
        Args:
            config_file: 配置文件路径，默认使用configs/anet_i3d.yaml
        """
        if config_file is None:
            config_file = os.path.join(
                os.path.dirname(__file__), 
                'configs', 
                'anet_i3d.yaml'
            )
        
        # 加载配置
        self.cfg = load_config(config_file)
        print("配置加载成功!")
        print(f"数据集名称: {self.cfg['dataset_name']}")
        print(f"训练集: {self.cfg['train_split']}")
        print(f"验证集: {self.cfg['val_split']}")
    
    def create_dataset(self, is_training=True):
        """
        创建ActivityNet数据集实例
        
        Args:
            is_training: 是否为训练模式
            
        Returns:
            dataset: 数据集实例
        """
        split = self.cfg['train_split'] if is_training else self.cfg['val_split']
        
        dataset = make_dataset(
            self.cfg['dataset_name'], 
            is_training, 
            split, 
            **self.cfg['dataset']
        )
        
        print(f"\n{'训练' if is_training else '验证'}数据集创建成功!")
        print(f"  数据集大小: {len(dataset)}")
        print(f"  特征维度: {dataset.input_dim}")
        print(f"  类别数量: {dataset.num_classes}")
        
        return dataset
    
    def create_dataloader(self, dataset, is_training=True):
        """
        创建数据加载器
        
        Args:
            dataset: 数据集实例
            is_training: 是否为训练模式
            
        Returns:
            dataloader: 数据加载器
        """
        batch_size = self.cfg['loader']['batch_size'] if is_training else 1
        
        dataloader = make_data_loader(
            dataset, 
            is_training,
            None,  # generator
            batch_size, 
            self.cfg['loader']['num_workers']
        )
        
        print(f"数据加载器创建成功!")
        print(f"  Batch size: {batch_size}")
        print(f"  Num workers: {self.cfg['loader']['num_workers']}")
        
        return dataloader
    
    def inspect_data_sample(self, dataset, index=0):
        """
        检查数据样本的内容
        
        Args:
            dataset: 数据集实例
            index: 样本索引
        """
        sample = dataset[index]
        
        print(f"\n检查数据样本 #{index}:")
        print(f"  Video ID: {sample['video_id']}")
        print(f"  特征形状: {sample['feats'].shape}")  # (C, T)
        print(f"  视频时长: {sample['duration']:.2f} 秒")
        print(f"  FPS: {sample['fps']:.2f}")
        print(f"  特征步长: {sample['feat_stride']:.2f}")
        print(f"  每帧数量: {sample['feat_num_frames']:.2f}")
        
        if sample['segments'] is not None:
            print(f"  动作片段数量: {len(sample['segments'])}")
            print(f"  片段时间 (特征帧):")
            for i, (seg, label) in enumerate(zip(sample['segments'], sample['labels'])):
                start_sec = seg[0].item() * sample['feat_stride'] / sample['fps']
                end_sec = seg[1].item() * sample['feat_stride'] / sample['fps']
                print(f"    [{i}] 标签={label.item()}, 帧=[{seg[0].item():.1f}, {seg[1].item():.1f}], 秒=[{start_sec:.2f}, {end_sec:.2f}]")
        else:
            print(f"  无标注信息")
        
        return sample
    
    def create_evaluator(self, dataset):
        """
        创建评估器
        
        Args:
            dataset: 验证数据集
            
        Returns:
            evaluator: 评估器实例
        """
        db_vars = dataset.get_attributes()
        
        evaluator = ANETdetection(
            dataset.json_file,
            dataset.split[0],
            tiou_thresholds=db_vars['tiou_thresholds']
        )
        
        print("\n评估器创建成功!")
        print(f"  tIoU阈值: {db_vars['tiou_thresholds']}")
        
        return evaluator
    
    def evaluate_predictions(self, evaluator, predictions):
        """
        评估预测结果
        
        Args:
            evaluator: 评估器
            predictions: 预测结果字典，包含:
                - 'video-id': 视频ID列表
                - 't-start': 开始时间列表
                - 't-end': 结束时间列表
                - 'label': 标签列表
                - 'score': 置信度分数列表
                
        Returns:
            mAP, average_mAP, mRecall: 评估指标
        """
        print("\n开始评估预测结果...")
        
        # 评估
        mAP, average_mAP, mRecall = evaluator.evaluate(predictions, verbose=True)
        
        return mAP, average_mAP, mRecall


def prepare_activitynet_data():
    """
    准备ActivityNet数据的完整流程说明
    """
    print("\n" + "="*80)
    print("ActivityNet数据集准备指南")
    print("="*80)
    
    instructions = """
1. 下载ActivityNet 1.3数据集
   官方网站: http://activity-net.org/download.html
   
   需要下载的文件:
   - 视频文件 (可选，如果需要重新提取特征)
   - 标注文件: activity_net.v1-3.min.json

2. 提取视频特征 (如使用I3D)
   特征提取可以使用多种方法:
   - I3D (Inflated 3D ConvNet)
   - TSP (Two-Stream Pooling)
   - 或其他视频特征提取器
   
   特征应保存为 .npy 格式，每个视频一个文件

3. 组织数据目录结构:
   data/
   └── anet_1.3/
       ├── annotations/
       │   ├── anet1.3_i3d_filtered.json  # 标注文件
       │   └── cuhk_val_simp_share.json   # 外部分类器分数(可选)
       └── i3d_features/                  # 特征文件夹
           ├── v_video1.npy
           ├── v_video2.npy
           └── ...

4. 标注文件格式:
   {
     "database": {
       "video_id": {
         "subset": "training",  # 或 "validation" 或 "testing"
         "duration": 120.5,     # 视频时长(秒)
         "fps": 25,             # 帧率
         "annotations": [
           {
             "segment": [10.2, 25.8],  # 动作片段 [开始, 结束] (秒)
             "label": "Doing gymnastics",
             "label_id": 10
           },
           ...
         ]
       },
       ...
     }
   }

5. 配置文件设置:
   在 configs/anet_i3d.yaml 中设置:
   - json_file: 标注文件路径
   - feat_folder: 特征文件夹路径
   - file_prefix: 特征文件前缀 (如 "v_")
   - file_ext: 特征文件扩展名 (如 ".npy")
   - input_dim: 特征维度 (如 I3D 为 2048)
   - num_classes: 类别数量
"""
    print(instructions)


def create_dummy_predictions():
    """
    创建示例预测结果（用于演示评估流程）
    """
    # 这是一个示例，实际使用时应该是模型的预测结果
    predictions = {
        'video-id': ['v_video1', 'v_video1', 'v_video2'],
        't-start': np.array([10.0, 50.0, 20.0]),
        't-end': np.array([25.0, 70.0, 45.0]),
        'label': np.array([0, 5, 10]),
        'score': np.array([0.95, 0.87, 0.92])
    }
    return predictions


def main():
    """主函数：展示完整的使用流程"""
    
    print("\n" + "="*80)
    print("ActivityNet数据集使用示例")
    print("="*80)
    
    # 1. 显示数据准备指南
    prepare_activitynet_data()
    
    # 2. 检查配置文件是否存在
    config_file = os.path.join(
        os.path.dirname(__file__), 
        'configs', 
        'anet_i3d.yaml'
    )
    
    if not os.path.exists(config_file):
        print(f"\n错误: 配置文件不存在: {config_file}")
        print("请先创建配置文件或使用正确的路径")
        return
    
    try:
        # 3. 创建数据集示例实例
        example = ActivityNetDatasetExample(config_file)
        
        # 4. 检查数据文件是否存在
        json_file = example.cfg['dataset']['json_file']
        feat_folder = example.cfg['dataset']['feat_folder']
        
        if not os.path.exists(json_file):
            print(f"\n警告: 标注文件不存在: {json_file}")
            print("请按照上述指南准备数据")
            return
        
        if not os.path.exists(feat_folder):
            print(f"\n警告: 特征文件夹不存在: {feat_folder}")
            print("请按照上述指南准备数据")
            return
        
        # 5. 创建训练和验证数据集
        print("\n" + "-"*80)
        print("创建训练数据集...")
        print("-"*80)
        train_dataset = example.create_dataset(is_training=True)
        
        print("\n" + "-"*80)
        print("创建验证数据集...")
        print("-"*80)
        val_dataset = example.create_dataset(is_training=False)
        
        # 6. 创建数据加载器
        print("\n" + "-"*80)
        print("创建数据加载器...")
        print("-"*80)
        train_loader = example.create_dataloader(train_dataset, is_training=True)
        val_loader = example.create_dataloader(val_dataset, is_training=False)
        
        # 7. 检查数据样本
        print("\n" + "-"*80)
        print("检查训练集样本...")
        print("-"*80)
        if len(train_dataset) > 0:
            example.inspect_data_sample(train_dataset, index=0)
        
        # 8. 创建评估器
        print("\n" + "-"*80)
        print("创建评估器...")
        print("-"*80)
        evaluator = example.create_evaluator(val_dataset)
        
        # 9. 演示评估流程 (使用虚拟预测)
        print("\n" + "-"*80)
        print("演示评估流程 (使用示例预测)...")
        print("-"*80)
        print("注意: 这只是演示，实际使用时需要模型的真实预测结果")
        
        # dummy_predictions = create_dummy_predictions()
        # mAP, avg_mAP, mRecall = example.evaluate_predictions(evaluator, dummy_predictions)
        
        print("\n" + "="*80)
        print("示例完成!")
        print("="*80)
        print("\n使用提示:")
        print("1. 按照上述指南准备ActivityNet数据")
        print("2. 修改配置文件中的路径")
        print("3. 使用 train.py 进行训练")
        print("4. 使用 eval.py 进行评估")
        
    except Exception as e:
        print(f"\n错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == '__main__':
    main()

