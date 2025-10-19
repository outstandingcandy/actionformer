#!/usr/bin/env python3
"""
分析训练集和验证集的loss变化
"""

import os
import json
import sys
import glob

def load_train_losses(train_eval_dir):
    """从训练集评估结果中加载loss"""
    summary_file = os.path.join(train_eval_dir, 'all_train_eval_results.json')
    if not os.path.exists(summary_file):
        return {}
    
    with open(summary_file, 'r') as f:
        all_results = json.load(f)
    
    train_losses = {}
    for epoch_key, results in all_results.items():
        epoch = results.get('epoch', 0)
        # 训练集评估时没有loss，需要从训练日志获取
        train_losses[epoch] = results
    
    return train_losses

def load_val_losses(eval_dir):
    """从验证集评估结果中加载loss"""
    val_losses = {}
    
    # 查找所有验证结果文件
    result_files = glob.glob(os.path.join(eval_dir, 'eval_results_epoch_*.json'))
    
    for result_file in sorted(result_files):
        with open(result_file, 'r') as f:
            results = json.load(f)
        
        # 从文件名提取epoch
        filename = os.path.basename(result_file)
        epoch = int(filename.split('_')[-1].replace('.json', ''))
        
        if 'validation_loss' in results and results['validation_loss']:
            val_losses[epoch] = results['validation_loss']
            
            # 同时保存mAP信息
            if 'results' in results:
                val_losses[epoch]['mAP'] = results['results'].get('mAP', 0.0)
                val_losses[epoch]['mAP@0.5'] = results['results'].get('mAP@0.5', 0.0)
    
    return val_losses

def analyze_losses(exp_dir):
    """分析一个实验的训练和验证loss"""
    
    train_eval_dir = os.path.join(exp_dir, 'train_eval_results')
    eval_dir = os.path.join(exp_dir, 'eval_results')
    
    exp_name = os.path.basename(exp_dir)
    
    print("=" * 80)
    print(f"实验: {exp_name}")
    print("=" * 80)
    
    # 加载验证集loss
    val_losses = load_val_losses(eval_dir)
    
    if not val_losses:
        print("⚠️ 未找到验证集loss数据")
        return
    
    print(f"\n找到 {len(val_losses)} 个epoch的验证集loss")
    
    # 打印详细信息
    print("\n" + "-" * 80)
    print(f"{'Epoch':<8} {'Val Loss':<12} {'Val Cls':<12} {'Val Reg':<12} {'mAP@0.5':<12} {'mAP':<12}")
    print("-" * 80)
    
    epochs = sorted(val_losses.keys())
    for epoch in epochs:
        loss_data = val_losses[epoch]
        final_loss = loss_data.get('final_loss', 0.0)
        cls_loss = loss_data.get('cls_loss', 0.0)
        reg_loss = loss_data.get('reg_loss', 0.0)
        map05 = loss_data.get('mAP@0.5', 0.0)
        map_avg = loss_data.get('mAP', 0.0)
        
        print(f"{epoch:<8} {final_loss:<12.4f} {cls_loss:<12.4f} {reg_loss:<12.4f} {map05:<12.4f} {map_avg:<12.4f}")
    
    # 统计信息
    if len(epochs) > 1:
        first_epoch = epochs[0]
        last_epoch = epochs[-1]
        
        first_loss = val_losses[first_epoch]['final_loss']
        last_loss = val_losses[last_epoch]['final_loss']
        loss_change = ((last_loss - first_loss) / first_loss) * 100
        
        first_map = val_losses[first_epoch].get('mAP', 0.0)
        last_map = val_losses[last_epoch].get('mAP', 0.0)
        
        print("\n" + "=" * 80)
        print("统计摘要:")
        print("=" * 80)
        print(f"Loss变化:")
        print(f"  初始 (Epoch {first_epoch}): {first_loss:.4f}")
        print(f"  最终 (Epoch {last_epoch}): {last_loss:.4f}")
        print(f"  变化: {loss_change:+.1f}%")
        
        # 找到最低loss
        min_loss_epoch = min(epochs, key=lambda e: val_losses[e]['final_loss'])
        min_loss = val_losses[min_loss_epoch]['final_loss']
        print(f"  最低 (Epoch {min_loss_epoch}): {min_loss:.4f}")
        
        print(f"\nmAP变化:")
        print(f"  初始 (Epoch {first_epoch}): {first_map:.4f}")
        print(f"  最终 (Epoch {last_epoch}): {last_map:.4f}")
        
        # 找到最高mAP
        max_map_epoch = max(epochs, key=lambda e: val_losses[e].get('mAP', 0.0))
        max_map = val_losses[max_map_epoch].get('mAP', 0.0)
        print(f"  最高 (Epoch {max_map_epoch}): {max_map:.4f}")
        
        # 过拟合分析
        print(f"\n过拟合分析:")
        if min_loss_epoch != max_map_epoch:
            print(f"  ⚠️ 最低loss的epoch ({min_loss_epoch}) ≠ 最高mAP的epoch ({max_map_epoch})")
            print(f"  这可能表明存在轻微过拟合")
        else:
            print(f"  ✓ 最低loss和最高mAP在同一epoch ({min_loss_epoch})")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='分析训练和验证loss')
    parser.add_argument('--exp_dir', type=str, 
                        help='实验目录路径（例如: ckpt/badminton_swin_test_swin）')
    parser.add_argument('--list', action='store_true',
                        help='列出所有可用的实验')
    
    args = parser.parse_args()
    
    if args.list:
        # 列出所有实验
        ckpt_dir = 'ckpt'
        if os.path.exists(ckpt_dir):
            exps = [d for d in os.listdir(ckpt_dir) if os.path.isdir(os.path.join(ckpt_dir, d))]
            print("可用的实验:")
            for exp in sorted(exps):
                exp_path = os.path.join(ckpt_dir, exp)
                eval_dir = os.path.join(exp_path, 'eval_results')
                if os.path.exists(eval_dir):
                    num_results = len(glob.glob(os.path.join(eval_dir, 'eval_results_epoch_*.json')))
                    print(f"  - {exp} ({num_results} epochs)")
        return
    
    if args.exp_dir:
        analyze_losses(args.exp_dir)
    else:
        # 默认分析最新的实验
        ckpt_dir = 'ckpt'
        if os.path.exists(ckpt_dir):
            exps = [os.path.join(ckpt_dir, d) for d in os.listdir(ckpt_dir) 
                   if os.path.isdir(os.path.join(ckpt_dir, d)) and 
                   os.path.exists(os.path.join(ckpt_dir, d, 'eval_results'))]
            
            if exps:
                # 按修改时间排序，取最新的
                latest_exp = max(exps, key=lambda x: os.path.getmtime(x))
                print(f"分析最新实验: {os.path.basename(latest_exp)}\n")
                analyze_losses(latest_exp)
            else:
                print("未找到实验目录。使用 --exp_dir 指定实验目录，或 --list 列出所有实验。")

if __name__ == '__main__':
    main()

