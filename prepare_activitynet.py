#!/usr/bin/env python3
"""
ActivityNet数据准备和验证工具

功能：
1. 验证标注文件格式
2. 检查特征文件完整性
3. 生成数据统计信息
4. 转换标注格式
"""

import os
import sys
import json
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm


class ActivityNetDataPreparer:
    """ActivityNet数据准备工具类"""
    
    def __init__(self, annotation_file, feature_dir=None):
        """
        初始化
        
        Args:
            annotation_file: 标注文件路径
            feature_dir: 特征文件目录 (可选)
        """
        self.annotation_file = annotation_file
        self.feature_dir = feature_dir
        self.data = None
        
        if os.path.exists(annotation_file):
            self.load_annotations()
        else:
            print(f"警告: 标注文件不存在: {annotation_file}")
    
    def load_annotations(self):
        """加载标注文件"""
        print(f"加载标注文件: {self.annotation_file}")
        with open(self.annotation_file, 'r', encoding='utf-8') as f:
            self.data = json.load(f)
        
        if 'database' not in self.data:
            raise ValueError("标注文件格式错误: 缺少 'database' 字段")
        
        print(f"  版本: {self.data.get('version', 'unknown')}")
        print(f"  视频数量: {len(self.data['database'])}")
    
    def validate_annotations(self):
        """验证标注文件格式"""
        print("\n" + "="*80)
        print("验证标注文件格式...")
        print("="*80)
        
        if self.data is None:
            print("错误: 未加载标注数据")
            return False
        
        database = self.data['database']
        total_videos = len(database)
        
        errors = []
        warnings = []
        
        for video_id, video_info in tqdm(database.items(), desc="验证"):
            # 检查必需字段
            if 'subset' not in video_info:
                errors.append(f"{video_id}: 缺少 'subset' 字段")
            elif video_info['subset'] not in ['training', 'validation', 'testing']:
                errors.append(f"{video_id}: 'subset' 值无效: {video_info['subset']}")
            
            if 'duration' not in video_info:
                errors.append(f"{video_id}: 缺少 'duration' 字段")
            elif not isinstance(video_info['duration'], (int, float)):
                errors.append(f"{video_id}: 'duration' 类型错误")
            elif video_info['duration'] <= 0:
                errors.append(f"{video_id}: 'duration' 值无效: {video_info['duration']}")
            
            # 检查标注
            if 'annotations' in video_info:
                annotations = video_info['annotations']
                if not isinstance(annotations, list):
                    errors.append(f"{video_id}: 'annotations' 应该是列表")
                else:
                    for idx, ann in enumerate(annotations):
                        # 检查segment
                        if 'segment' not in ann:
                            errors.append(f"{video_id} 标注#{idx}: 缺少 'segment'")
                        elif len(ann['segment']) != 2:
                            errors.append(f"{video_id} 标注#{idx}: 'segment' 应该有2个元素")
                        else:
                            start, end = ann['segment']
                            if start >= end:
                                errors.append(f"{video_id} 标注#{idx}: segment 开始时间 >= 结束时间")
                            if start < 0:
                                errors.append(f"{video_id} 标注#{idx}: segment 开始时间 < 0")
                            if end > video_info.get('duration', float('inf')):
                                warnings.append(f"{video_id} 标注#{idx}: segment 结束时间超过视频时长")
                        
                        # 检查label
                        if 'label' not in ann:
                            errors.append(f"{video_id} 标注#{idx}: 缺少 'label'")
                        
                        # 检查label_id
                        if 'label_id' not in ann:
                            warnings.append(f"{video_id} 标注#{idx}: 缺少 'label_id'")
                        elif not isinstance(ann['label_id'], int):
                            errors.append(f"{video_id} 标注#{idx}: 'label_id' 应该是整数")
            else:
                if video_info.get('subset') != 'testing':
                    warnings.append(f"{video_id}: 训练/验证集视频缺少标注")
        
        # 输出结果
        print(f"\n验证完成:")
        print(f"  总视频数: {total_videos}")
        print(f"  错误数量: {len(errors)}")
        print(f"  警告数量: {len(warnings)}")
        
        if errors:
            print("\n错误列表 (显示前10个):")
            for error in errors[:10]:
                print(f"  ❌ {error}")
            if len(errors) > 10:
                print(f"  ... 还有 {len(errors)-10} 个错误")
        
        if warnings:
            print("\n警告列表 (显示前10个):")
            for warning in warnings[:10]:
                print(f"  ⚠️  {warning}")
            if len(warnings) > 10:
                print(f"  ... 还有 {len(warnings)-10} 个警告")
        
        if not errors:
            print("\n✅ 标注文件格式验证通过!")
            return True
        else:
            print("\n❌ 标注文件存在错误，请修正")
            return False
    
    def check_features(self, file_prefix='v_', file_ext='.npy'):
        """检查特征文件完整性"""
        print("\n" + "="*80)
        print("检查特征文件...")
        print("="*80)
        
        if self.feature_dir is None:
            print("未指定特征目录")
            return
        
        if not os.path.exists(self.feature_dir):
            print(f"特征目录不存在: {self.feature_dir}")
            return
        
        if self.data is None:
            print("未加载标注数据")
            return
        
        database = self.data['database']
        
        missing_features = []
        existing_features = []
        feature_stats = {
            'shapes': defaultdict(int),
            'dtypes': defaultdict(int),
        }
        
        print(f"特征目录: {self.feature_dir}")
        print(f"文件前缀: {file_prefix}")
        print(f"文件扩展名: {file_ext}")
        
        for video_id in tqdm(database.keys(), desc="检查特征"):
            feature_file = os.path.join(
                self.feature_dir,
                f"{file_prefix}{video_id}{file_ext}"
            )
            
            if os.path.exists(feature_file):
                existing_features.append(video_id)
                
                # 读取特征信息
                try:
                    features = np.load(feature_file)
                    feature_stats['shapes'][features.shape] += 1
                    feature_stats['dtypes'][str(features.dtype)] += 1
                except Exception as e:
                    print(f"\n警告: 无法加载特征 {video_id}: {e}")
            else:
                missing_features.append(video_id)
        
        # 输出统计
        print(f"\n特征文件统计:")
        print(f"  总视频数: {len(database)}")
        print(f"  已有特征: {len(existing_features)} ({len(existing_features)/len(database)*100:.1f}%)")
        print(f"  缺失特征: {len(missing_features)} ({len(missing_features)/len(database)*100:.1f}%)")
        
        if feature_stats['shapes']:
            print(f"\n特征形状分布:")
            for shape, count in sorted(feature_stats['shapes'].items(), key=lambda x: -x[1])[:10]:
                print(f"    {shape}: {count} 个文件")
        
        if feature_stats['dtypes']:
            print(f"\n特征数据类型:")
            for dtype, count in feature_stats['dtypes'].items():
                print(f"    {dtype}: {count} 个文件")
        
        if missing_features:
            print(f"\n缺失特征列表 (显示前20个):")
            for video_id in missing_features[:20]:
                print(f"    {video_id}")
            if len(missing_features) > 20:
                print(f"    ... 还有 {len(missing_features)-20} 个")
            
            # 保存缺失列表
            missing_file = os.path.join(
                os.path.dirname(self.annotation_file),
                'missing_features.txt'
            )
            with open(missing_file, 'w') as f:
                for video_id in missing_features:
                    f.write(f"{video_id}\n")
            print(f"\n缺失特征列表已保存到: {missing_file}")
    
    def generate_statistics(self):
        """生成数据统计信息"""
        print("\n" + "="*80)
        print("数据统计信息")
        print("="*80)
        
        if self.data is None:
            print("未加载标注数据")
            return
        
        database = self.data['database']
        
        # 基本统计
        stats = {
            'total_videos': len(database),
            'subsets': defaultdict(int),
            'total_annotations': 0,
            'videos_with_annotations': 0,
            'durations': [],
            'fps_values': [],
            'annotations_per_video': [],
            'segment_durations': [],
            'labels': defaultdict(int),
        }
        
        for video_id, video_info in database.items():
            # 子集统计
            subset = video_info.get('subset', 'unknown')
            stats['subsets'][subset] += 1
            
            # 时长统计
            if 'duration' in video_info:
                stats['durations'].append(video_info['duration'])
            
            # FPS统计
            if 'fps' in video_info:
                stats['fps_values'].append(video_info['fps'])
            
            # 标注统计
            if 'annotations' in video_info and len(video_info['annotations']) > 0:
                stats['videos_with_annotations'] += 1
                num_annotations = len(video_info['annotations'])
                stats['annotations_per_video'].append(num_annotations)
                stats['total_annotations'] += num_annotations
                
                for ann in video_info['annotations']:
                    # 片段时长统计
                    if 'segment' in ann and len(ann['segment']) == 2:
                        duration = ann['segment'][1] - ann['segment'][0]
                        stats['segment_durations'].append(duration)
                    
                    # 标签统计
                    if 'label' in ann:
                        stats['labels'][ann['label']] += 1
        
        # 输出统计
        print(f"\n基本信息:")
        print(f"  总视频数: {stats['total_videos']}")
        print(f"  总标注数: {stats['total_annotations']}")
        print(f"  有标注的视频: {stats['videos_with_annotations']}")
        
        print(f"\n子集分布:")
        for subset, count in sorted(stats['subsets'].items()):
            print(f"  {subset}: {count} ({count/stats['total_videos']*100:.1f}%)")
        
        if stats['durations']:
            durations = np.array(stats['durations'])
            print(f"\n视频时长统计 (秒):")
            print(f"  平均: {durations.mean():.2f}")
            print(f"  中位数: {np.median(durations):.2f}")
            print(f"  最小: {durations.min():.2f}")
            print(f"  最大: {durations.max():.2f}")
            print(f"  标准差: {durations.std():.2f}")
        
        if stats['fps_values']:
            fps_values = np.array(stats['fps_values'])
            print(f"\nFPS统计:")
            print(f"  平均: {fps_values.mean():.2f}")
            print(f"  中位数: {np.median(fps_values):.2f}")
            print(f"  范围: [{fps_values.min():.2f}, {fps_values.max():.2f}]")
        
        if stats['annotations_per_video']:
            ann_per_vid = np.array(stats['annotations_per_video'])
            print(f"\n每视频标注数统计:")
            print(f"  平均: {ann_per_vid.mean():.2f}")
            print(f"  中位数: {np.median(ann_per_vid):.0f}")
            print(f"  最小: {ann_per_vid.min()}")
            print(f"  最大: {ann_per_vid.max()}")
        
        if stats['segment_durations']:
            seg_dur = np.array(stats['segment_durations'])
            print(f"\n动作片段时长统计 (秒):")
            print(f"  平均: {seg_dur.mean():.2f}")
            print(f"  中位数: {np.median(seg_dur):.2f}")
            print(f"  最小: {seg_dur.min():.2f}")
            print(f"  最大: {seg_dur.max():.2f}")
        
        print(f"\n类别统计:")
        print(f"  总类别数: {len(stats['labels'])}")
        print(f"  Top 10 类别:")
        sorted_labels = sorted(stats['labels'].items(), key=lambda x: -x[1])
        for label, count in sorted_labels[:10]:
            print(f"    {label}: {count}")
        
        # 保存统计信息
        stats_file = os.path.join(
            os.path.dirname(self.annotation_file),
            'dataset_statistics.json'
        )
        
        # 转换为可序列化的格式
        serializable_stats = {
            'total_videos': stats['total_videos'],
            'total_annotations': stats['total_annotations'],
            'videos_with_annotations': stats['videos_with_annotations'],
            'subsets': dict(stats['subsets']),
            'num_classes': len(stats['labels']),
            'duration_stats': {
                'mean': float(np.mean(stats['durations'])) if stats['durations'] else 0,
                'median': float(np.median(stats['durations'])) if stats['durations'] else 0,
                'min': float(np.min(stats['durations'])) if stats['durations'] else 0,
                'max': float(np.max(stats['durations'])) if stats['durations'] else 0,
            },
            'top_10_labels': sorted_labels[:10]
        }
        
        with open(stats_file, 'w', encoding='utf-8') as f:
            json.dump(serializable_stats, f, indent=2, ensure_ascii=False)
        
        print(f"\n统计信息已保存到: {stats_file}")
    
    def filter_annotations(self, output_file, min_duration=1.0, remove_duplicates=True):
        """
        过滤和清理标注
        
        Args:
            output_file: 输出文件路径
            min_duration: 最小片段时长(秒)
            remove_duplicates: 是否删除重复标注
        """
        print("\n" + "="*80)
        print("过滤标注...")
        print("="*80)
        
        if self.data is None:
            print("未加载标注数据")
            return
        
        print(f"过滤参数:")
        print(f"  最小片段时长: {min_duration} 秒")
        print(f"  删除重复标注: {remove_duplicates}")
        
        filtered_data = {
            'version': self.data.get('version', '1.3'),
            'database': {}
        }
        
        removed_count = 0
        
        for video_id, video_info in tqdm(self.data['database'].items(), desc="过滤"):
            filtered_video = video_info.copy()
            
            if 'annotations' in video_info:
                filtered_annotations = []
                
                for ann in video_info['annotations']:
                    # 检查片段时长
                    if 'segment' in ann and len(ann['segment']) == 2:
                        duration = ann['segment'][1] - ann['segment'][0]
                        if duration < min_duration:
                            removed_count += 1
                            continue
                    
                    # 检查重复
                    if remove_duplicates:
                        is_duplicate = False
                        for existing_ann in filtered_annotations:
                            if (abs(ann['segment'][0] - existing_ann['segment'][0]) < 0.001 and
                                abs(ann['segment'][1] - existing_ann['segment'][1]) < 0.001 and
                                ann.get('label_id') == existing_ann.get('label_id')):
                                is_duplicate = True
                                removed_count += 1
                                break
                        
                        if is_duplicate:
                            continue
                    
                    filtered_annotations.append(ann)
                
                filtered_video['annotations'] = filtered_annotations
            
            filtered_data['database'][video_id] = filtered_video
        
        # 保存过滤后的数据
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, indent=2, ensure_ascii=False)
        
        print(f"\n过滤完成:")
        print(f"  移除的标注数: {removed_count}")
        print(f"  输出文件: {output_file}")


def main():
    parser = argparse.ArgumentParser(
        description='ActivityNet数据准备和验证工具'
    )
    parser.add_argument('annotation_file', type=str,
                        help='标注文件路径')
    parser.add_argument('--feature-dir', type=str, default=None,
                        help='特征文件目录')
    parser.add_argument('--file-prefix', type=str, default='v_',
                        help='特征文件前缀')
    parser.add_argument('--file-ext', type=str, default='.npy',
                        help='特征文件扩展名')
    parser.add_argument('--validate', action='store_true',
                        help='验证标注文件格式')
    parser.add_argument('--check-features', action='store_true',
                        help='检查特征文件完整性')
    parser.add_argument('--statistics', action='store_true',
                        help='生成数据统计信息')
    parser.add_argument('--filter', type=str, default=None,
                        help='过滤标注并保存到指定文件')
    parser.add_argument('--min-duration', type=float, default=1.0,
                        help='最小片段时长(秒)')
    parser.add_argument('--all', action='store_true',
                        help='执行所有检查')
    
    args = parser.parse_args()
    
    # 创建准备器
    preparer = ActivityNetDataPreparer(
        args.annotation_file,
        args.feature_dir
    )
    
    # 执行操作
    if args.all:
        preparer.validate_annotations()
        if args.feature_dir:
            preparer.check_features(args.file_prefix, args.file_ext)
        preparer.generate_statistics()
    else:
        if args.validate:
            preparer.validate_annotations()
        
        if args.check_features:
            if args.feature_dir is None:
                print("错误: 需要指定 --feature-dir")
            else:
                preparer.check_features(args.file_prefix, args.file_ext)
        
        if args.statistics:
            preparer.generate_statistics()
    
    if args.filter:
        preparer.filter_annotations(
            args.filter,
            min_duration=args.min_duration
        )


if __name__ == '__main__':
    main()

