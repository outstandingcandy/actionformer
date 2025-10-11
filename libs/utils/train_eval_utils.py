"""
训练集内部评估工具
"""

import os
import json
import numpy as np
import pandas as pd
import torch
from .badminton_metrics import BadmintonDetection
from .badminton_validation import organize_predictions_by_video

def evaluate_on_training_set(model, train_loader, evaluator, epoch, output_dir, device='cuda', max_batches=None):
    """
    在训练集上进行评估
    
    Args:
        model: 训练好的模型
        train_loader: 训练数据加载器
        evaluator: 评估器
        epoch: 当前epoch
        output_dir: 输出目录
        device: 设备
        max_batches: 最大评估batch数量，None表示评估全部数据
    """
    if max_batches is not None:
        print(f"\n=== 训练集内部评估 (Epoch {epoch}) - 只评估前 {max_batches} 个batch ===")
    else:
        print(f"\n=== 训练集内部评估 (Epoch {epoch}) ===")
    
    model.eval()
    all_predictions = []
    
    with torch.no_grad():
        for batch_idx, batch_data in enumerate(train_loader):
            # 如果设置了max_batches，达到限制后停止
            if max_batches is not None and batch_idx >= max_batches:
                print(f"  已达到评估限制 ({max_batches} batches)，停止评估")
                break
            
            # batch_data是一个列表，包含多个字典
            # 逐个处理每个样本，因为模型在推理模式下只支持batch_size=1
            for i, data_dict in enumerate(batch_data):
                # 将数据移到设备
                for key in data_dict:
                    if isinstance(data_dict[key], torch.Tensor):
                        data_dict[key] = data_dict[key].to(device)
                
                # 前向传播 - 单个样本
                single_batch = [data_dict]
                outputs = model(single_batch)
                
                # 处理预测结果
                predictions = process_predictions(outputs, single_batch)
                all_predictions.extend(predictions)
            
            if batch_idx % 10 == 0 or (max_batches is not None and batch_idx == max_batches - 1):
                total_batches = max_batches if max_batches is not None else len(train_loader)
                print(f"  处理批次 {batch_idx + 1}/{total_batches}")
    
    # 转换为DataFrame
    if all_predictions:
        pred_df = pd.DataFrame(all_predictions)
        print(f"  生成预测数量: {len(pred_df)}")
        
        # 运行评估
        results = evaluator.evaluate(pred_df, verbose=False)
        
        # 添加详细预测结果（与验证集格式一致）
        detailed_predictions = organize_predictions_by_video(pred_df, evaluator)
        results['detailed_predictions'] = detailed_predictions
    else:
        print("  警告: 没有生成任何预测")
        # 创建空结果
        results = {
            'label': evaluator.action_names.values(),
            '0.1': [0.0] * len(evaluator.action_names),
            '0.2': [0.0] * len(evaluator.action_names),
            '0.3': [0.0] * len(evaluator.action_names),
            '0.4': [0.0] * len(evaluator.action_names),
            '0.5': [0.0] * len(evaluator.action_names),
            'detailed_predictions': {}
        }
    
    # 保存结果
    save_train_eval_results(results, epoch, output_dir)
    
    # 打印结果
    print_train_eval_results(results, epoch)
    
    model.train()
    return results

def process_predictions(outputs, batch_data):
    """
    处理模型输出，转换为评估格式
    
    Args:
        outputs: 模型输出
        batch_data: 批次数据
        
    Returns:
        predictions: 预测列表
    """
    predictions = []
    
    # 处理模型输出 - 输出是一个列表，包含一个字典
    if isinstance(outputs, list) and len(outputs) > 0:
        output_dict = outputs[0]  # 获取第一个（也是唯一的）输出
        
        if 'video_id' in output_dict and 'segments' in output_dict and 'scores' in output_dict and 'labels' in output_dict:
            video_id = output_dict['video_id']
            segments = output_dict['segments']  # [N, 2]
            scores = output_dict['scores']      # [N]
            labels = output_dict['labels']      # [N]
            
            # 过滤低置信度预测
            valid_mask = scores > 0.005  # 使用配置中的min_score
            
            for i in range(len(valid_mask)):
                if valid_mask[i]:
                    pred = {
                        'video-id': video_id,
                        't-start': float(segments[i][0]),
                        't-end': float(segments[i][1]),
                        'label': int(labels[i]),
                        'score': float(scores[i])
                    }
                    predictions.append(pred)
    
    return predictions

def save_train_eval_results(results, epoch, output_dir):
    """
    保存训练集评估结果
    
    Args:
        results: 评估结果
        epoch: 当前epoch
        output_dir: 输出目录
    """
    # 创建输出目录
    eval_dir = os.path.join(output_dir, 'train_eval_results')
    os.makedirs(eval_dir, exist_ok=True)
    
    # 添加格式化的结果统计（与验证集格式保持一致）
    formatted_results = {
        'epoch': epoch,
        'results': {},
        'AP_per_class': [],
        'detailed_predictions': results.get('detailed_predictions', {})
    }
    
    # 计算各种mAP指标
    if 'label' in results:
        labels = results['label']
        
        # 计算不同IoU阈值的mAP
        for tiou in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if tiou in results:
                ap_values = results[tiou]
                map_val = np.mean(ap_values)
                formatted_results['results'][f'mAP@{tiou}'] = float(map_val)
        
        # 计算整体mAP (所有IoU阈值的平均)
        map_values = []
        for tiou in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if tiou in results:
                map_values.append(np.mean(results[tiou]))
        if map_values:
            formatted_results['results']['mAP'] = float(np.mean(map_values))
        
        # 保存每个类别的AP (0.5 IoU)
        if 0.5 in results:
            formatted_results['AP_per_class'] = [float(ap) for ap in results[0.5]]
            
            # 添加类别名称和AP的映射
            class_performance = {}
            for i, label in enumerate(labels):
                if i < len(results[0.5]):
                    class_performance[label] = float(results[0.5][i])
            formatted_results['results']['AP_by_class'] = class_performance
        
        # 添加预测统计
        if 'detailed_predictions' in results:
            total_predictions = sum(
                video_data['num_predictions'] 
                for video_data in results['detailed_predictions'].values()
            )
            formatted_results['results']['total_predictions'] = total_predictions
            formatted_results['results']['num_videos'] = len(results['detailed_predictions'])
    
    # 保存当前epoch的结果
    epoch_file = os.path.join(eval_dir, f'train_eval_epoch_{epoch:03d}.json')
    with open(epoch_file, 'w', encoding='utf-8') as f:
        json.dump(formatted_results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"  训练集评估结果已保存到: {epoch_file}")
    
    # 更新所有结果文件（不包含detailed_predictions以节省空间）
    all_results_file = os.path.join(eval_dir, 'all_train_eval_results.json')
    
    # 创建不含详细预测的摘要版本
    summary_results = {
        'epoch': epoch,
        'results': formatted_results['results'],
        'AP_per_class': formatted_results['AP_per_class']
    }
    
    if os.path.exists(all_results_file):
        with open(all_results_file, 'r', encoding='utf-8') as f:
            all_results = json.load(f)
    else:
        all_results = {}
    
    all_results[f'epoch_{epoch:03d}'] = summary_results
    
    with open(all_results_file, 'w', encoding='utf-8') as f:
        json.dump(all_results, f, indent=2, default=str, ensure_ascii=False)
    
    print(f"  汇总结果已更新到: {all_results_file}")
    
    # 保存简化的统计摘要
    summary_file = os.path.join(eval_dir, 'train_eval_summary.txt')
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write(f"\n=== Epoch {epoch} ===\n")
        
        if 'label' in results:
            # 计算mAP
            for tiou in [0.1, 0.2, 0.3, 0.4, 0.5]:
                if tiou in results:
                    ap_values = results[tiou]
                    map_val = np.mean(ap_values)
                    f.write(f"mAP@{tiou:.1f}: {map_val:.4f}\n")
            
            # 整体mAP
            map_values = []
            for tiou in [0.1, 0.2, 0.3, 0.4, 0.5]:
                if tiou in results:
                    ap_values = results[tiou]
                    map_values.append(np.mean(ap_values))
            
            if map_values:
                overall_map = np.mean(map_values)
                f.write(f"整体mAP: {overall_map:.4f}\n")
            
            # 各类别性能
            f.write("\n各动作类别性能 (AP@0.5):\n")
            labels = results['label']
            for i, label in enumerate(labels):
                if 0.5 in results and i < len(results[0.5]):
                    ap_val = results[0.5][i]
                    f.write(f"  {label}: {ap_val:.4f}\n")
    
    print(f"  训练集评估摘要已追加到: {summary_file}")

def print_train_eval_results(results, epoch):
    """
    打印训练集评估结果
    
    Args:
        results: 评估结果
        epoch: 当前epoch
    """
    print(f"\n训练集评估结果 (Epoch {epoch}):")
    
    # 计算mAP
    if 'label' in results:
        labels = results['label']
        
        # 计算不同IoU阈值下的mAP
        map_values = []
        for tiou in [0.1, 0.2, 0.3, 0.4, 0.5]:
            if tiou in results:
                ap_values = results[tiou]
                map_val = np.mean(ap_values)
                map_values.append(map_val)
                print(f"  mAP@{tiou:.1f}: {map_val:.4f}")
        
        if map_values:
            overall_map = np.mean(map_values)
            print(f"  整体mAP: {overall_map:.4f}")
        
        # 显示各类别性能
        print("\n各动作类别性能:")
        for i, label in enumerate(labels):
            if 0.5 in results and i < len(results[0.5]):
                ap_val = results[0.5][i]
                print(f"    {label}: {ap_val:.4f}")

def create_train_evaluator(config, train_dataset=None):
    """
    创建训练集评估器
    
    Args:
        config: 配置字典
        train_dataset: 训练数据集（可选，用于获取标签字典）
        
    Returns:
        evaluator: 评估器
    """
    json_file = config['dataset']['json_file']
    
    evaluator = BadmintonDetection(
        ant_file=json_file,
        split='training',  # 使用训练集
        tiou_thresholds=np.linspace(0.1, 0.5, 5)
    )
    
    # 如果提供了训练数据集，使用数据集的标签字典
    if train_dataset is not None and hasattr(train_dataset, 'label_dict'):
        evaluator.action_names = train_dataset.label_dict
    
    return evaluator
