#!/usr/bin/env python3
"""
ActivityNet 推理程序
使用 Uniformer 特征提取器 + ActionFormer 模型

支持两种模式：
1. 从视频文件提取特征并推理
2. 从已有特征文件推理
"""

import os
import sys
import json
import argparse
import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import pickle

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from libs.core import load_config
from libs.modeling import make_meta_arch
from libs.datasets import make_dataset, make_data_loader
from libs.utils import ANETdetection
from extract_features import SimpleFeatureExtractor


class ActivityNetInference:
    """ActivityNet 推理类"""
    
    def __init__(
        self,
        config_file,
        checkpoint_file,
        annotation_file,
        feature_dir=None,
        video_dir=None,
        output_dir='./inference_results',
        device='cuda',
        use_pretrained=False,
        ignore_keys=None
    ):
        """
        初始化推理器
        
        Args:
            config_file: 配置文件路径
            checkpoint_file: 模型 checkpoint 路径
            annotation_file: 标注文件路径
            feature_dir: 特征文件目录（如果已有特征）
            video_dir: 视频文件目录（如果需要提取特征）
            output_dir: 输出目录
            device: 设备 (cuda/cpu)
            use_pretrained: 是否使用预训练模型（宽松加载模式）
            ignore_keys: 忽略的键名前缀列表
        """
        self.config_file = config_file
        self.checkpoint_file = checkpoint_file
        self.annotation_file = annotation_file
        self.feature_dir = feature_dir
        self.video_dir = video_dir
        self.output_dir = output_dir
        self.device = device
        self.use_pretrained = use_pretrained
        self.ignore_keys = ignore_keys or []
        
        os.makedirs(output_dir, exist_ok=True)
        
        # 加载配置
        print("="*80)
        print("加载配置文件...")
        print("="*80)
        self.cfg = load_config(config_file)
        print(f"配置: {config_file}")
        print(f"Checkpoint: {checkpoint_file}")
        print(f"标注文件: {annotation_file}")
        
        # 加载模型
        print("\n加载模型...")
        self.model = self._load_model()
        
        # 特征提取器（如果需要）
        self.feature_extractor = None
        if video_dir is not None and feature_dir is None:
            print("\n初始化特征提取器...")
            self.feature_extractor = SimpleFeatureExtractor(
                device=device,
                target_fps=self.cfg['dataset'].get('default_fps', 25.0),
                feature_type='uniformer'
            )
    
    def _load_model(self):
        """加载 ActionFormer 模型"""
        # 创建模型
        model = make_meta_arch(self.cfg['model_name'], **self.cfg['model'])
        
        # 加载权重
        print(f"从 checkpoint 加载权重: {self.checkpoint_file}")
        checkpoint = torch.load(
            self.checkpoint_file,
            map_location=lambda storage, loc: storage.cuda(
                torch.cuda.current_device() if torch.cuda.is_available() else 'cpu'
            )
        )
        
        # 尝试加载 EMA 模型或普通模型
        if 'state_dict_ema' in checkpoint:
            print("✓ 检测到 EMA 模型权重")
            state_dict = checkpoint['state_dict_ema']
        elif 'state_dict' in checkpoint:
            print("✓ 检测到标准模型权重")
            state_dict = checkpoint['state_dict']
        else:
            print("✓ 使用直接权重字典")
            state_dict = checkpoint
        
        # 显示 checkpoint 信息（如果有）
        if 'epoch' in checkpoint:
            print(f"  - Epoch: {checkpoint['epoch']}")
        if 'best_mAP' in checkpoint:
            print(f"  - Best mAP: {checkpoint['best_mAP']:.4f}")
        
        # 处理 DataParallel 的 state_dict
        new_state_dict = {}
        has_module_prefix = False
        for k, v in state_dict.items():
            if k.startswith('module.'):
                has_module_prefix = True
                new_state_dict[k[7:]] = v  # 移除 'module.' 前缀
            else:
                new_state_dict[k] = v
        
        if has_module_prefix:
            print("  - 已移除 'module.' 前缀（DataParallel 格式）")
        
        # 如果指定了忽略的键，移除它们
        if self.ignore_keys:
            filtered_state_dict = {}
            ignored_count = 0
            for k, v in new_state_dict.items():
                should_ignore = any(k.startswith(prefix) for prefix in self.ignore_keys)
                if not should_ignore:
                    filtered_state_dict[k] = v
                else:
                    ignored_count += 1
            new_state_dict = filtered_state_dict
            if ignored_count > 0:
                print(f"  - 已忽略 {ignored_count} 个键（根据 --ignore-keys）")
                print(f"    忽略前缀: {self.ignore_keys}")
        
        # 加载权重
        strict_mode = not self.use_pretrained  # 预训练模型使用宽松模式
        
        try:
            if strict_mode:
                model.load_state_dict(new_state_dict, strict=True)
                print("  ✓ 严格模式加载成功（所有权重完全匹配）")
            else:
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                print("  ✓ 宽松模式加载（适用于预训练模型）")
                if missing_keys:
                    print(f"    - 缺失的键 ({len(missing_keys)} 个): {missing_keys[:5]}")
                    if len(missing_keys) > 5:
                        print(f"      ... 还有 {len(missing_keys)-5} 个")
                if unexpected_keys:
                    print(f"    - 意外的键 ({len(unexpected_keys)} 个): {unexpected_keys[:5]}")
                    if len(unexpected_keys) > 5:
                        print(f"      ... 还有 {len(unexpected_keys)-5} 个")
        except RuntimeError as e:
            if strict_mode:
                print(f"⚠️  严格加载失败，切换到宽松模式...")
                print(f"     错误: {str(e)[:100]}")
                missing_keys, unexpected_keys = model.load_state_dict(new_state_dict, strict=False)
                if missing_keys:
                    print(f"  - 缺失的键 ({len(missing_keys)} 个): {missing_keys[:5]}")
                if unexpected_keys:
                    print(f"  - 意外的键 ({len(unexpected_keys)} 个): {unexpected_keys[:5]}")
                print("  - 已使用宽松模式加载模型")
            else:
                raise e
        
        # 设置为评估模式
        model.eval()
        model = model.to(self.device)
        
        print(f"✅ 模型加载成功")
        return model
    
    def extract_features_from_video(self, video_path):
        """
        从视频提取特征
        
        Args:
            video_path: 视频路径
            
        Returns:
            features: 特征数组 (T, C)
            fps: 视频帧率
        """
        if self.feature_extractor is None:
            raise ValueError("特征提取器未初始化")
        
        print(f"从视频提取特征: {video_path}")
        features, fps = self.feature_extractor.extract_video_features(video_path)
        return features, fps
    
    def load_features_from_file(self, feature_file):
        """
        从文件加载特征
        
        Args:
            feature_file: 特征文件路径
            
        Returns:
            features: 特征数组 (T, C)
        """
        if not os.path.exists(feature_file):
            raise FileNotFoundError(f"特征文件不存在: {feature_file}")
        
        features = np.load(feature_file).astype(np.float32)
        return features
    
    def prepare_input(self, features, video_info):
        """
        准备模型输入
        
        Args:
            features: 特征数组 (T, C)
            video_info: 视频信息字典
            
        Returns:
            data_dict: 模型输入字典
        """
        # 获取视频信息
        fps = video_info.get('fps', self.cfg['dataset'].get('default_fps', 25.0))
        duration = video_info.get('duration', features.shape[0] / fps)
        
        # 特征参数
        feat_stride = self.cfg['dataset'].get('feat_stride', 16)
        num_frames = self.cfg['dataset'].get('num_frames', 16)
        max_seq_len = self.cfg['dataset'].get('max_seq_len', 192)
        
        # 转换为 PyTorch 张量 (C, T)
        feats = torch.from_numpy(features.T).float()
        
        # 上采样到固定长度（如果需要）
        if self.cfg['dataset'].get('force_upsampling', False):
            if feats.shape[-1] != max_seq_len:
                feats = torch.nn.functional.interpolate(
                    feats.unsqueeze(0),
                    size=max_seq_len,
                    mode='linear',
                    align_corners=False
                ).squeeze(0)
        
        # 计算 feat_stride
        if self.cfg['dataset'].get('force_upsampling', False):
            feat_stride = float(
                (features.shape[0] - 1) * feat_stride + num_frames
            ) / max_seq_len
        
        feat_offset = 0.5 * num_frames / feat_stride
        
        # 构建数据字典
        data_dict = {
            'video_id': video_info.get('id', 'unknown'),
            'feats': feats.unsqueeze(0).to(self.device),  # (1, C, T)
            'fps': fps,
            'duration': duration,
            'feat_stride': feat_stride,
            'feat_num_frames': num_frames
        }
        
        return data_dict
    
    def inference_single_video(self, data_dict):
        """
        对单个视频进行推理
        
        Args:
            data_dict: 输入数据字典
            
        Returns:
            results: 预测结果
        """
        with torch.no_grad():
            # 模型推理
            output = self.model(data_dict)
            
            # 提取预测结果
            results = {
                'video_id': data_dict['video_id'],
                'segments': [],
                'labels': [],
                'scores': []
            }
            
            # 处理模型输出
            if len(output) > 0 and 'segments' in output[0]:
                pred_segments = output[0]['segments'].cpu().numpy()  # (N, 2)
                pred_labels = output[0]['labels'].cpu().numpy()      # (N,)
                pred_scores = output[0]['scores'].cpu().numpy()      # (N,)
                
                # 转换特征帧到时间（秒）
                fps = data_dict['fps']
                feat_stride = data_dict['feat_stride']
                feat_offset = 0.5 * data_dict['feat_num_frames'] / feat_stride
                
                for seg, label, score in zip(pred_segments, pred_labels, pred_scores):
                    # 转换为秒
                    start_sec = (seg[0] + feat_offset) * feat_stride / fps
                    end_sec = (seg[1] + feat_offset) * feat_stride / fps
                    
                    results['segments'].append([float(start_sec), float(end_sec)])
                    results['labels'].append(int(label))
                    results['scores'].append(float(score))
            
            return results
    
    def run_inference(self, video_list=None, split='validation', max_videos=None):
        """
        运行推理
        
        Args:
            video_list: 视频ID列表（如果为None，使用所有视频）
            split: 数据集划分 (training/validation/testing)
            max_videos: 最大处理视频数（用于测试）
            
        Returns:
            all_results: 所有视频的预测结果
        """
        print("\n" + "="*80)
        print("开始推理...")
        print("="*80)
        
        # 加载标注文件
        with open(self.annotation_file, 'r') as f:
            annotations = json.load(f)
        
        database = annotations.get('database', annotations)
        
        # 筛选视频
        if video_list is not None:
            videos_to_process = {vid: info for vid, info in database.items() 
                                if vid in video_list}
        else:
            videos_to_process = {vid: info for vid, info in database.items()
                                if info.get('subset', '').lower() == split.lower()}
        
        if max_videos:
            videos_to_process = dict(list(videos_to_process.items())[:max_videos])
        
        print(f"待处理视频数: {len(videos_to_process)}")
        print(f"数据集划分: {split}")
        
        # 推理所有视频
        all_results = []
        failed_videos = []
        
        for video_id, video_info in tqdm(videos_to_process.items(), desc="推理进度"):
            try:
                # 准备特征
                if self.feature_dir:
                    # 从特征文件加载
                    feature_file = os.path.join(
                        self.feature_dir,
                        f"{self.cfg['dataset'].get('file_prefix', '')}{video_id}{self.cfg['dataset'].get('file_ext', '.npy')}"
                    )
                    features = self.load_features_from_file(feature_file)
                    video_info['fps'] = video_info.get('fps', self.cfg['dataset'].get('default_fps', 25.0))
                
                elif self.video_dir:
                    # 从视频提取特征
                    video_file = os.path.join(self.video_dir, f"{video_id}.mp4")
                    if not os.path.exists(video_file):
                        print(f"\n警告: 视频文件不存在: {video_file}")
                        failed_videos.append(video_id)
                        continue
                    features, fps = self.extract_features_from_video(video_file)
                    video_info['fps'] = fps
                
                else:
                    raise ValueError("必须提供 feature_dir 或 video_dir")
                
                # 准备输入
                video_info['id'] = video_id
                data_dict = self.prepare_input(features, video_info)
                
                # 推理
                results = self.inference_single_video(data_dict)
                all_results.append(results)
                
            except Exception as e:
                print(f"\n错误: 处理视频 {video_id} 时出错: {e}")
                failed_videos.append(video_id)
                continue
        
        # 打印摘要
        print("\n" + "="*80)
        print("推理完成")
        print("="*80)
        print(f"成功: {len(all_results)}/{len(videos_to_process)}")
        print(f"失败: {len(failed_videos)}")
        
        if failed_videos:
            print(f"\n失败的视频 (前10个):")
            for vid in failed_videos[:10]:
                print(f"  - {vid}")
        
        return all_results
    
    def save_results(self, results, output_file=None):
        """
        保存推理结果
        
        Args:
            results: 推理结果列表
            output_file: 输出文件路径
        """
        if output_file is None:
            output_file = os.path.join(self.output_dir, 'predictions.json')
        
        print(f"\n保存结果到: {output_file}")
        
        # 转换为 JSON 格式
        output_data = {
            'version': 'ActivityNet-v1.3',
            'results': {},
            'external_data': {}
        }
        
        for result in results:
            video_id = result['video_id']
            predictions = []
            
            for seg, label, score in zip(
                result['segments'],
                result['labels'],
                result['scores']
            ):
                predictions.append({
                    'segment': seg,
                    'label': label,
                    'score': score
                })
            
            output_data['results'][video_id] = predictions
        
        with open(output_file, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"✅ 结果已保存")
        
        # 同时保存为 pickle 格式（用于评估）
        pkl_file = output_file.replace('.json', '.pkl')
        
        # 转换为评估器需要的格式
        eval_results = {
            'video-id': [],
            't-start': [],
            't-end': [],
            'label': [],
            'score': []
        }
        
        for result in results:
            for seg, label, score in zip(
                result['segments'],
                result['labels'],
                result['scores']
            ):
                eval_results['video-id'].append(result['video_id'])
                eval_results['t-start'].append(seg[0])
                eval_results['t-end'].append(seg[1])
                eval_results['label'].append(label)
                eval_results['score'].append(score)
        
        with open(pkl_file, 'wb') as f:
            pickle.dump(eval_results, f)
        
        print(f"✅ Pickle 格式已保存: {pkl_file}")
    
    def evaluate_results(self, results, split='validation'):
        """
        评估推理结果
        
        Args:
            results: 推理结果
            split: 数据集划分
            
        Returns:
            eval_metrics: 评估指标
        """
        print("\n" + "="*80)
        print("评估结果...")
        print("="*80)
        
        # 创建评估器
        evaluator = ANETdetection(
            self.annotation_file,
            split=split,
            tiou_thresholds=np.linspace(0.5, 0.95, 10)
        )
        
        # 准备预测数据
        predictions = {
            'video-id': [],
            't-start': [],
            't-end': [],
            'label': [],
            'score': []
        }
        
        for result in results:
            for seg, label, score in zip(
                result['segments'],
                result['labels'],
                result['scores']
            ):
                predictions['video-id'].append(result['video_id'])
                predictions['t-start'].append(seg[0])
                predictions['t-end'].append(seg[1])
                predictions['label'].append(label)
                predictions['score'].append(score)
        
        # 评估
        if len(predictions['video-id']) > 0:
            predictions['t-start'] = np.array(predictions['t-start'])
            predictions['t-end'] = np.array(predictions['t-end'])
            predictions['label'] = np.array(predictions['label'])
            predictions['score'] = np.array(predictions['score'])
            
            mAP, avg_mAP, mRecall = evaluator.evaluate(predictions, verbose=True)
            
            return {
                'mAP': mAP,
                'average_mAP': avg_mAP,
                'mRecall': mRecall
            }
        else:
            print("⚠️  没有预测结果，跳过评估")
            return None


def main():
    parser = argparse.ArgumentParser(
        description='ActivityNet 推理程序 (Uniformer + ActionFormer)',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

1. 使用已有特征推理:
   python3 inference_activitynet.py \\
       --config configs/anet_uniformer.yaml \\
       --checkpoint ckpt/anet_uniformer/epoch_010.pth.tar \\
       --annotation data/activitynet_hf/activity_net.v1-3.min.json \\
       --feature-dir data/activitynet_features \\
       --split validation

2. 从视频提取特征并推理:
   python3 inference_activitynet.py \\
       --config configs/anet_uniformer.yaml \\
       --checkpoint ckpt/anet_uniformer/epoch_010.pth.tar \\
       --annotation data/activitynet_hf/activity_net.v1-3.min.json \\
       --video-dir data/activitynet_videos \\
       --split validation

3. 测试模式（只处理前10个视频）:
   python3 inference_activitynet.py \\
       --config configs/anet_uniformer.yaml \\
       --checkpoint ckpt/anet_uniformer/epoch_010.pth.tar \\
       --annotation data/activitynet_hf/activity_net.v1-3.min.json \\
       --feature-dir data/activitynet_features \\
       --max-videos 10
        """
    )
    
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='模型 checkpoint 路径')
    parser.add_argument('--annotation', type=str, required=True,
                        help='标注文件路径')
    parser.add_argument('--feature-dir', type=str, default=None,
                        help='特征文件目录')
    parser.add_argument('--video-dir', type=str, default=None,
                        help='视频文件目录')
    parser.add_argument('--output-dir', type=str, default='./inference_results',
                        help='输出目录')
    parser.add_argument('--split', type=str, default='validation',
                        choices=['training', 'validation', 'testing'],
                        help='数据集划分')
    parser.add_argument('--max-videos', type=int, default=None,
                        help='最大处理视频数（测试用）')
    parser.add_argument('--no-eval', action='store_true',
                        help='不进行评估')
    parser.add_argument('--device', type=str, default='cuda',
                        help='设备 (cuda/cpu)')
    parser.add_argument('--use-pretrained', action='store_true',
                        help='使用官方预训练模型（允许部分权重不匹配）')
    parser.add_argument('--ignore-keys', type=str, nargs='*', default=[],
                        help='加载权重时忽略的键名前缀（用于迁移学习）')
    
    args = parser.parse_args()
    
    # 检查参数
    if args.feature_dir is None and args.video_dir is None:
        parser.error("必须提供 --feature-dir 或 --video-dir 之一")
    
    # 创建推理器
    inferencer = ActivityNetInference(
        config_file=args.config,
        checkpoint_file=args.checkpoint,
        annotation_file=args.annotation,
        feature_dir=args.feature_dir,
        video_dir=args.video_dir,
        output_dir=args.output_dir,
        device=args.device,
        use_pretrained=args.use_pretrained,
        ignore_keys=args.ignore_keys
    )
    
    # 运行推理
    results = inferencer.run_inference(
        split=args.split,
        max_videos=args.max_videos
    )
    
    # 保存结果
    inferencer.save_results(results)
    
    # 评估（如果不是测试集）
    if not args.no_eval and args.split != 'testing':
        eval_metrics = inferencer.evaluate_results(results, split=args.split)
        
        # 保存评估指标
        if eval_metrics:
            metrics_file = os.path.join(args.output_dir, 'eval_metrics.json')
            with open(metrics_file, 'w') as f:
                json.dump({
                    'average_mAP': float(eval_metrics['average_mAP']),
                    'mAP_per_iou': eval_metrics['mAP'].tolist()
                }, f, indent=2)
            print(f"\n✅ 评估指标已保存: {metrics_file}")
    
    print("\n" + "="*80)
    print("所有任务完成！")
    print("="*80)
    print(f"结果保存在: {args.output_dir}")


if __name__ == '__main__':
    main()

