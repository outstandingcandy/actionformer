#!/usr/bin/env python3
"""
从保存的checkpoint计算验证集loss
"""

import os
import sys
import json
import torch
import argparse
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))

from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import AverageMeter

def compute_validation_loss(model, val_loader, device='cuda'):
    """计算验证集loss"""
    
    model.eval()
    losses_tracker = {}
    
    print("计算验证集loss...")
    
    with torch.no_grad():
        for batch_idx, video_list in enumerate(tqdm(val_loader, desc="处理验证集")):
            # 临时切换到训练模式以计算loss
            model.train()
            losses = model(video_list)
            model.eval()
            
            # 追踪loss
            for key, value in losses.items():
                if key not in losses_tracker:
                    losses_tracker[key] = AverageMeter()
                losses_tracker[key].update(value.item())
    
    # 打印结果
    print("\n验证集Loss统计:")
    print("-" * 50)
    for key, value in losses_tracker.items():
        print(f"  {key}: {value.avg:.4f}")
    
    return {key: value.avg for key, value in losses_tracker.items()}

def main():
    parser = argparse.ArgumentParser(description='从checkpoint计算验证集loss')
    parser.add_argument('--config', type=str, required=True,
                        help='配置文件路径')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='checkpoint文件路径')
    parser.add_argument('--output', type=str, default=None,
                        help='输出JSON文件路径（可选）')
    
    args = parser.parse_args()
    
    # 加载配置
    cfg = load_config(args.config)
    print(f"配置文件: {args.config}")
    print(f"Checkpoint: {args.checkpoint}")
    
    # 创建验证集
    val_dataset = make_dataset(
        cfg['dataset_name'],
        False,
        cfg['val_split'],
        **cfg['dataset']
    )
    
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers']
    )
    
    print(f"验证集样本数: {len(val_dataset)}")
    
    # 创建模型
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    
    # 加载checkpoint
    print(f"\n加载checkpoint...")
    checkpoint = torch.load(args.checkpoint, map_location='cpu')
    
    # 处理DataParallel的state_dict
    state_dict = checkpoint.get('state_dict_ema', checkpoint.get('state_dict'))
    if state_dict is None:
        print("错误: checkpoint中未找到模型参数")
        return
    
    # 如果是DataParallel保存的，需要去掉'module.'前缀
    from collections import OrderedDict
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k.replace('module.', '') if k.startswith('module.') else k
        new_state_dict[name] = v
    
    model.load_state_dict(new_state_dict)
    print("✓ Checkpoint加载成功")
    
    # 移动到GPU
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = torch.nn.DataParallel(model, device_ids=cfg['devices'])
    model.to(device)
    
    # 计算验证集loss
    val_losses = compute_validation_loss(model, val_loader, device)
    
    # 保存结果
    if args.output:
        output_data = {
            'checkpoint': args.checkpoint,
            'epoch': checkpoint.get('epoch', 0),
            'validation_loss': val_losses
        }
        
        with open(args.output, 'w') as f:
            json.dump(output_data, f, indent=2)
        
        print(f"\n结果已保存到: {args.output}")
    
    return val_losses

if __name__ == '__main__':
    main()

