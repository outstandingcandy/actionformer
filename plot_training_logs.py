#!/usr/bin/env python3
"""
绘制训练日志中的loss和mAP曲线
"""

import json
import os
import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_training_curves_from_logs(log_file_path, output_dir=None, show_plot=True):
    """
    从训练日志文件绘制曲线
    
    Args:
        log_file_path: 日志文件路径 (JSON格式)
        output_dir: 输出目录，如果为None则使用日志文件所在目录
        show_plot: 是否显示图表
    """
    print(f"📊 正在读取日志文件: {log_file_path}")
    
    try:
        with open(log_file_path, 'r', encoding='utf-8') as f:
            all_eval_results = json.load(f)
    except Exception as e:
        print(f"❌ 读取日志文件失败: {e}")
        return
    
    if not all_eval_results:
        print("❌ 日志文件为空或无评估数据")
        return
    
    print(f"✅ 成功读取 {len(all_eval_results)} 个epoch的评估数据")
    
    # 设置输出目录
    if output_dir is None:
        output_dir = os.path.dirname(log_file_path)
    os.makedirs(output_dir, exist_ok=True)
    
    # 确保中文显示
    plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans', 'Arial Unicode MS', 'Noto Sans CJK']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 设置seaborn样式
    sns.set_style("whitegrid")
    
    # 准备数据
    epochs = []
    train_losses = []
    val_losses = []
    train_map = []
    val_map = []
    val_class_ap = {}  # 类别AP数据
    
    # 检查是否为单个epoch结果文件格式
    if 'results' in all_eval_results and 'AP_per_class' in all_eval_results.get('results', {}):
        # 单个epoch结果文件格式，从文件名提取epoch号
        filename = os.path.basename(log_file_path)
        try:
            epoch_num = int(filename.split('_')[-1].replace('.json', ''))
            epochs = [epoch_num]
            
            # 处理单个epoch的数据
            results = all_eval_results
            
            # 训练loss (如果有的话)
            train_losses = [results.get('train_loss', None)]
                
            # 验证loss
            if 'validation_loss' in results:
                val_loss = results['validation_loss'].get('final_loss', 0.0)
                val_losses = [val_loss]
            else:
                val_losses = [None]
                
            # mAP数据
            if 'results' in results:
                map_data = results['results']
                train_map = [map_data.get('train_mAP@0.5', 0.0) * 100]
                val_map = [map_data.get('mAP@0.5', 0.0) * 100]
                
                # 提取类别AP
                if 'AP_per_class' in map_data:
                    # 对于单个epoch，我们需要类别名称
                    # 这里假设有12个类别，按顺序对应
                    class_names = ['勾球', '发球', '吊球', '扑球', '抽球', '挑球', 
                                  '挡网', '推球', '放网前球', '杀球', '高远球', '未知']
                    ap_values = map_data['AP_per_class']
                    for i, ap_value in enumerate(ap_values):
                        if i < len(class_names):
                            class_name = class_names[i]
                            if class_name not in val_class_ap:
                                val_class_ap[class_name] = []
                            val_class_ap[class_name].append(ap_value * 100)
        except (ValueError, IndexError) as e:
            print(f"❌ 无法从文件名解析epoch号: {filename}")
            return
    else:
        # 多epoch结果文件格式 (all_eval_results.json)
        for epoch_key, results in all_eval_results.items():
            try:
                # 尝试不同的epoch_key格式
                if 'epoch_' in epoch_key:
                    epoch_num = int(epoch_key.split('_')[1])
                elif epoch_key.isdigit():
                    epoch_num = int(epoch_key)
                else:
                    # 跳过无法解析的key
                    print(f"⚠️ 跳过无法解析的epoch_key: {epoch_key}")
                    continue
                epochs.append(epoch_num)
            except (ValueError, IndexError) as e:
                print(f"⚠️ 解析epoch_key失败: {epoch_key}, 错误: {e}")
                continue
            
            # 训练loss (如果有的话)
            if 'train_loss' in results:
                train_losses.append(results['train_loss'])
            else:
                train_losses.append(None)
                
            # 验证loss
            if 'validation_loss' in results:
                val_loss = results['validation_loss'].get('final_loss', 0.0)
                val_losses.append(val_loss)
            else:
                val_losses.append(None)
                
            # mAP数据
            if 'results' in results:
                map_data = results['results']
                train_map.append(map_data.get('train_mAP@0.5', 0.0) * 100)  # 转换为百分比
                val_map.append(map_data.get('mAP@0.5', 0.0) * 100)
                
                # 提取类别AP (使用最后一次评估的数据)
                if 'AP_by_class' in map_data:
                    for class_name, ap_value in map_data['AP_by_class'].items():
                        if class_name not in val_class_ap:
                            val_class_ap[class_name] = []
                        val_class_ap[class_name].append(ap_value * 100)
    
    if not epochs:
        print("❌ 没有评估数据可用于绘图")
        return
    
    # 创建图表
    fig = plt.figure(figsize=(16, 12))
    
    # 1. Loss曲线图
    plt.subplot(2, 2, 1)
    if any(x is not None for x in train_losses):
        plt.plot(epochs, train_losses, 'b-', label='训练Loss', linewidth=2, marker='o', markersize=4)
    if any(x is not None for x in val_losses):
        plt.plot(epochs, val_losses, 'r-', label='验证Loss', linewidth=2, marker='s', markersize=4)
    
    plt.title('训练和验证Loss曲线', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('Loss', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 2. mAP曲线图
    plt.subplot(2, 2, 2)
    if any(x > 0 for x in train_map):
        plt.plot(epochs, train_map, 'g-', label='训练mAP@0.5', linewidth=2, marker='o', markersize=4)
    if any(x > 0 for x in val_map):
        plt.plot(epochs, val_map, 'purple', label='验证mAP@0.5', linewidth=2, marker='s', markersize=4)
    
    plt.title('训练和验证mAP@0.5曲线', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch', fontsize=12)
    plt.ylabel('mAP@0.5 (%)', fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # 3. 各动作类别AP表现 (柱状图)
    plt.subplot(2, 2, 3)
    if val_class_ap:
        # 使用最后一次评估的数据
        last_epoch_ap = {}
        for class_name, ap_list in val_class_ap.items():
            if ap_list:
                last_epoch_ap[class_name] = ap_list[-1]
        
        if last_epoch_ap:
            classes = list(last_epoch_ap.keys())
            ap_values = list(last_epoch_ap.values())
            
            # 按AP值排序
            sorted_pairs = sorted(zip(classes, ap_values), key=lambda x: x[1], reverse=True)
            classes_sorted, ap_values_sorted = zip(*sorted_pairs)
            
            # 创建柱状图
            colors = plt.cm.viridis(np.linspace(0, 1, len(classes_sorted)))
            bars = plt.bar(range(len(classes_sorted)), ap_values_sorted, color=colors, alpha=0.7)
            
            plt.title('各动作类别AP@0.5表现', fontsize=14, fontweight='bold')
            plt.xlabel('动作类别', fontsize=12)
            plt.ylabel('AP@0.5 (%)', fontsize=12)
            plt.xticks(range(len(classes_sorted)), classes_sorted, rotation=45, ha='right')
            plt.grid(True, alpha=0.3, axis='y')
            
            # 在柱子上标注数值
            for i, bar in enumerate(bars):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height,
                        f'{height:.1f}%', ha='center', va='bottom', fontsize=9)
    
    # 4. 训练统计信息
    plt.subplot(2, 2, 4)
    plt.axis('off')
    
    # 计算统计信息
    max_epoch = max(epochs) if epochs else 0
    best_val_map = max(val_map) if val_map else 0


    best_train_map = max(train_map) if train_map else 0
    final_train_map = train_map[-1] if train_map else 0
    final_val_map = val_map[-1] if val_map else 0
    
    # Top动作
    top_action = "无数据"
    if val_class_ap and best_val_map > 0:
        try:
            top_action_name = max(val_class_ap.items(), key=lambda x: x[1][-1] if x[1] else 0)[0]
            top_action_value = max(val_class_ap.items(), key=lambda x: x[1][-1] if x[1] else 0)[1][-1]
            top_action = f"{top_action_name}: {top_action_value:.1f}%"
        except:
            top_action = "无数据"
    
    stats_text = f"""
📊 训练统计信息

🎯 训练概要:
   总Epochs: {max_epoch}
   数据集: 羽毛球动作检测

📈 性能表现:
   训练mAP@0.5: {final_train_map:.1f}% (最高: {best_train_map:.1f}%)
   验证mAP@0.5: {final_val_map:.1f}% (最高: {best_val_map:.1f}%)
   
📋 最佳验证性能:
   {'无数据' if best_val_map == 0 else f'{best_val_map:.1f}%'}
   
🏆 Top动作:
   {top_action}
"""
    
    plt.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.1))
    
    # 设置总标题
    config_name = os.path.basename(log_file_path).replace('.json', '').replace('_results', '')
    fig.suptitle(f'{config_name} - 训练曲线分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(output_dir, 'training_curves_analysis.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 训练曲线图已保存到: {plot_file}")
    
    # 同时保存为PDF格式
    plot_pdf_file = os.path.join(output_dir, 'training_curves_analysis.pdf')
    plt.savefig(plot_pdf_file, bbox_inches='tight', facecolor='white')
    print(f"📄 PDF版本已保存到: {plot_pdf_file}")
    
    if show_plot:
        plt.show()
    else:
        plt.close()
    
    print("✅ 训练曲线绘制完成!")

def main():
    parser = argparse.ArgumentParser(description='绘制训练日志中的loss和mAP曲线')
    parser.add_argument('log_file', help='训练日志JSON文件路径')
    parser.add_argument('-o', '--output', help='输出目录 (默认使用日志文件所在目录)')
    parser.add_argument('--no-show', action='store_true', help='不显示图表，仅保存文件')
    
    args = parser.parse_args()
    
    if not os.path.exists(args.log_file):
        print(f"❌ 日志文件不存在: {args.log_file}")
        return
    
    plot_training_curves_from_logs(
        args.log_file, 
        output_dir=args.output,
        show_plot=not args.no_show
    )

if __name__ == '__main__':
    main()
