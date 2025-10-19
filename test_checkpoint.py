#!/usr/bin/env python3
"""
Checkpoint 检查工具
用于验证和查看 checkpoint 文件的内容
"""

import torch
import argparse
import os


def inspect_checkpoint(checkpoint_path):
    """
    检查 checkpoint 文件
    
    Args:
        checkpoint_path: checkpoint 文件路径
    """
    print("="*80)
    print(f"检查 Checkpoint: {checkpoint_path}")
    print("="*80)
    
    if not os.path.exists(checkpoint_path):
        print(f"❌ 文件不存在: {checkpoint_path}")
        return
    
    # 加载 checkpoint
    print("\n加载 checkpoint...")
    try:
        checkpoint = torch.load(
            checkpoint_path,
            map_location='cpu'  # 加载到 CPU
        )
        print("✅ Checkpoint 加载成功\n")
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return
    
    # 检查 checkpoint 类型
    print("Checkpoint 类型:")
    if isinstance(checkpoint, dict):
        print("  ✓ 字典格式")
        
        # 显示顶层键
        print(f"\n顶层键 ({len(checkpoint)} 个):")
        for key in checkpoint.keys():
            print(f"  - {key}")
        
        # 检查训练信息
        print("\n训练信息:")
        if 'epoch' in checkpoint:
            print(f"  - Epoch: {checkpoint['epoch']}")
        if 'best_mAP' in checkpoint:
            print(f"  - Best mAP: {checkpoint['best_mAP']:.4f}")
        if 'optimizer' in checkpoint:
            print(f"  - 包含优化器状态")
        
        # 检查模型权重
        print("\n模型权重:")
        
        # 检查 EMA 权重
        if 'state_dict_ema' in checkpoint:
            print("  ✓ 包含 EMA 权重")
            state_dict = checkpoint['state_dict_ema']
            print(f"    - 参数数量: {len(state_dict)}")
            
            # 检查第一个参数
            first_key = list(state_dict.keys())[0]
            first_param = state_dict[first_key]
            print(f"    - 示例: {first_key}, shape={first_param.shape}")
        
        # 检查标准权重
        if 'state_dict' in checkpoint:
            print("  ✓ 包含标准权重")
            state_dict = checkpoint['state_dict']
            print(f"    - 参数数量: {len(state_dict)}")
            
            # 检查第一个参数
            first_key = list(state_dict.keys())[0]
            first_param = state_dict[first_key]
            print(f"    - 示例: {first_key}, shape={first_param.shape}")
        
        # 选择主要的 state_dict
        if 'state_dict_ema' in checkpoint:
            main_state_dict = checkpoint['state_dict_ema']
            print("\n使用 EMA 权重进行分析")
        elif 'state_dict' in checkpoint:
            main_state_dict = checkpoint['state_dict']
            print("\n使用标准权重进行分析")
        else:
            main_state_dict = checkpoint
            print("\n使用直接权重字典进行分析")
        
    else:
        print("  ⚠️  非字典格式（可能是旧版本格式）")
        main_state_dict = checkpoint
    
    # 分析权重结构
    print("\n权重结构分析:")
    print(f"  - 总参数数: {len(main_state_dict)}")
    
    # 检查 DataParallel 前缀
    has_module_prefix = any(k.startswith('module.') for k in main_state_dict.keys())
    if has_module_prefix:
        print("  ⚠️  包含 'module.' 前缀（DataParallel 格式）")
        print("     推理时会自动移除")
    else:
        print("  ✓ 标准格式（无 'module.' 前缀）")
    
    # 显示所有参数键（前20个）
    print(f"\n参数键列表（前20个）:")
    for i, key in enumerate(list(main_state_dict.keys())[:20]):
        param = main_state_dict[key]
        print(f"  {i+1:2d}. {key:50s} {str(param.shape):20s}")
    
    if len(main_state_dict) > 20:
        print(f"  ... 还有 {len(main_state_dict)-20} 个参数")
    
    # 计算总参数量
    total_params = sum(p.numel() for p in main_state_dict.values())
    print(f"\n总参数量: {total_params:,} ({total_params/1e6:.2f}M)")
    
    # 检查特定层（用于验证配置）
    print("\n关键层检查:")
    
    # 检查输入层
    input_keys = [k for k in main_state_dict.keys() if 'input' in k.lower() or 'embed' in k.lower()]
    if input_keys:
        print("  输入/嵌入层:")
        for key in input_keys[:3]:
            print(f"    - {key}: {main_state_dict[key].shape}")
    
    # 检查输出层
    output_keys = [k for k in main_state_dict.keys() if 'head' in k.lower() or 'fc' in k.lower() or 'classifier' in k.lower()]
    if output_keys:
        print("  输出层:")
        for key in output_keys[:3]:
            print(f"    - {key}: {main_state_dict[key].shape}")
    
    print("\n" + "="*80)
    print("检查完成")
    print("="*80)


def main():
    parser = argparse.ArgumentParser(
        description='检查和分析 PyTorch checkpoint 文件'
    )
    parser.add_argument('checkpoint', type=str,
                        help='checkpoint 文件路径')
    
    args = parser.parse_args()
    
    inspect_checkpoint(args.checkpoint)


if __name__ == '__main__':
    main()

