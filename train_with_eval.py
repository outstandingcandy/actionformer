# python imports
import argparse
import os
import time
import datetime
import json
from pprint import pprint

# torch imports
import torch
import torch.nn as nn
import torch.utils.data
# for visualization
from torch.utils.tensorboard import SummaryWriter

# our code
from libs.core import load_config
from libs.datasets import make_dataset, make_data_loader
from libs.modeling import make_meta_arch
from libs.utils import (train_one_epoch, valid_one_epoch, ANETdetection,
                        save_checkpoint, make_optimizer, make_scheduler,
                        fix_random_seed, ModelEma)
from libs.utils.badminton_metrics import BadmintonDetection
from libs.utils.badminton_validation import badminton_valid_one_epoch
from libs.utils.train_eval_utils import evaluate_on_training_set, create_train_evaluator
import numpy as np

# 绘图库导入
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.font_manager import FontProperties
import seaborn as sns
import pandas as pd

################################################################################
def plot_training_curves(all_eval_results, ckpt_folder, config_name):
    """
    绘制训练过程中的loss和mAP曲线
    """
    print("\n" + "="*60)
    print("📊 即将绘制训练曲线图表...")
    print("="*60)
    
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
    
    # 从结果中提取数据
    for epoch_key, results in all_eval_results.items():
        epoch_num = int(epoch_key.split('_')[1])
        epochs.append(epoch_num)
        
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
    
    stats_text = f"""
📊 训练统计信息

🎯 训练配置:
   模型: {config_name}
   总Epochs: {max_epoch}
   数据集: 羽毛球动作检测

📈 性能表现:
   训练mAP@0.5: {final_train_map:.1f}% (最高: {best_train_map:.1f}%)
   验证mAP@0.5: {final_val_map:.1f}% (最高: {best_val_map:.1f}%)
   
📋 最佳性能:
   {'无数据' if best_val_map == 0 else f'{best_val_map:.1f}%'}
   
🏆 Top动作:
   {max(val_class_ap.items(), key=lambda x: x[1][-1] if x[1] else 0)[0] + f': {max(val_class_ap.items(), key=lambda x: x[1][-1] if x[1] else 0)[1][-1]:.1f}%' if val_class_ap and best_val_map > 0 else '无数据'}
"""
    
    plt.text(0.1, 0.5, stats_text, fontsize=11, verticalalignment='center',
             bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray", alpha=0.1))
    
    # 设置总标题
    fig.suptitle(f'{config_name} - 训练曲线分析', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    # 保存图表
    plot_file = os.path.join(ckpt_folder, 'training_curves.png')
    plt.savefig(plot_file, dpi=300, bbox_inches='tight', facecolor='white')
    print(f"📊 训练曲线图已保存到: {plot_file}")
    
    # 同时保存为PDF格式
    plot_pdf_file = os.path.join(ckpt_folder, 'training_curves.pdf')
    plt.savefig(plot_pdf_file, bbox_inches='tight', facecolor='white')
    print(f"📄 PDF版本已保存到: {plot_pdf_file}")
    
    plt.close()
    print("✅ 训练曲线绘制完成!")

################################################################################
def main(args):
    """main function that handles training / inference with validation"""

    """1. setup parameters / folders"""
    # parse args
    args.start_epoch = 0
    if os.path.isfile(args.config):
        cfg = load_config(args.config)
    else:
        raise ValueError("Config file does not exist.")
    pprint(cfg)

    # prep for output folder (based on time stamp)
    if not os.path.exists(cfg['output_folder']):
        os.mkdir(cfg['output_folder'])
    cfg_filename = os.path.basename(args.config).replace('.yaml', '')
    if len(args.output) == 0:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(ts))
    else:
        ckpt_folder = os.path.join(
            cfg['output_folder'], cfg_filename + '_' + str(args.output))
    if not os.path.exists(ckpt_folder):
        os.mkdir(ckpt_folder)

    # create evaluation results folder
    eval_folder = os.path.join(ckpt_folder, 'eval_results')
    if not os.path.exists(eval_folder):
        os.mkdir(eval_folder)
        
    # tensorboard writer
    tb_writer = SummaryWriter(os.path.join(ckpt_folder, 'logs'))

    # fix the random seeds (this will fix everything)
    rng_generator = fix_random_seed(cfg['init_rand_seed'], include_cuda=True)

    # re-scale learning rate / # workers based on number of GPUs
    cfg['opt']["learning_rate"] *= len(cfg['devices'])
    cfg['loader']['num_workers'] *= len(cfg['devices'])

    """2. create dataset / dataloader"""
    train_dataset = make_dataset(
        cfg['dataset_name'], True, cfg['train_split'], **cfg['dataset']
    )
    # update cfg based on dataset attributes (fix to epic-kitchens)
    train_db_vars = train_dataset.get_attributes()
    cfg['model']['train_cfg']['head_empty_cls'] = train_db_vars['empty_label_ids']

    # data loaders
    train_loader = make_data_loader(
        train_dataset, True, rng_generator, **cfg['loader'])

    # validation dataset and loader
    val_dataset = make_dataset(
        cfg['dataset_name'], False, cfg['val_split'], **cfg['dataset']
    )
    val_loader = make_data_loader(
        val_dataset, False, None, 1, cfg['loader']['num_workers'])

    """3. create model, optimizer, and scheduler"""
    # model
    model = make_meta_arch(cfg['model_name'], **cfg['model'])
    # not ideal for multi GPU training, ok for now
    model = nn.DataParallel(model, device_ids=cfg['devices'])
    # optimizer
    optimizer = make_optimizer(model, cfg['opt'])
    # schedule
    num_iters_per_epoch = len(train_loader)
    scheduler = make_scheduler(optimizer, cfg['opt'], num_iters_per_epoch)

    # enable model EMA
    print("Using model EMA ...")
    model_ema = ModelEma(model)

    """4. Resume from model / Misc"""
    # resume from a checkpoint?
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(
                args.resume,
                map_location = lambda storage, loc: storage.cuda(
                    cfg['devices'][0]))
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            model_ema.module.load_state_dict(checkpoint['state_dict_ema'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            print("=> loaded checkpoint '{}' (epoch {})".format(
                args.resume, checkpoint['epoch']
            ))
            del checkpoint
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))
            return

    # save the current config
    with open(os.path.join(ckpt_folder, 'config.txt'), 'w') as fid:
        pprint(cfg, stream=fid)
        fid.flush()

    """5. training / validation loop"""
    print("\nStart training model {:s} ...".format(cfg['model_name']))
    
    # create evaluator for validation - 使用羽毛球专用的评估器
    val_db_vars = val_dataset.get_attributes()
    try:
        # 尝试使用标准的ANETdetection
        det_eval = ANETdetection(
            val_dataset.json_file,
            val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds']
        )
        print("Using ANETdetection evaluator")
    except KeyError:
        # 如果失败，使用羽毛球专用的评估器
        det_eval = BadmintonDetection(
            val_dataset.json_file,
            split=val_dataset.split[0],
            tiou_thresholds = val_db_vars['tiou_thresholds']
        )
        print("Using BadmintonDetection evaluator")
    
    # training loop with validation
    max_epochs = cfg['opt']['warmup_epochs'] + cfg['opt']['epochs']
    
    # store all evaluation results
    all_eval_results = {}
    
    # 创建训练集评估器
    train_evaluator = None
    if 'train_eval_split' in cfg:
        print("创建训练集评估器...")
        train_evaluator = create_train_evaluator(cfg, train_dataset)
    
    # 确定print_freq：优先使用配置文件中的值，其次使用命令行参数
    print_freq = cfg.get('print_freq', args.print_freq)
    print(f"Loss打印频率: 每 {print_freq} 个batch打印一次")
    
    for epoch in range(args.start_epoch, max_epochs):
        # train for one epoch
        train_one_epoch(
            train_loader,
            model,
            optimizer,
            scheduler,
            epoch,
            model_ema = model_ema,
            clip_grad_l2norm = cfg['train_cfg']['clip_grad_l2norm'],
            tb_writer=tb_writer,
            print_freq=print_freq
        )

        # 训练集内部评估
        if train_evaluator is not None and (epoch + 1) % args.eval_freq == 0:
            print(f"\n=> 训练集内部评估 at epoch {epoch + 1}")
            # 获取max_batches配置
            max_batches = cfg.get('train_eval_cfg', {}).get('max_batches', None)
            train_eval_results = evaluate_on_training_set(
                model_ema.module,
                train_loader,
                train_evaluator,
                epoch + 1,
                ckpt_folder,
                max_batches=max_batches
            )
            
            # 记录到tensorboard
            if 'label' in train_eval_results:
                for tiou in [0.1, 0.2, 0.3, 0.4, 0.5]:
                    if tiou in train_eval_results:
                        ap_values = train_eval_results[tiou]
                        map_val = np.mean(ap_values)
                        tb_writer.add_scalar(f'Train/mAP@{tiou:.1f}', map_val, epoch)

        # validation every few epochs
        if (epoch + 1) % args.eval_freq == 0 or (epoch + 1) == max_epochs:
            print(f"\n=> Evaluating at epoch {epoch + 1}")
            
            # validation output file for this epoch
            eval_file = os.path.join(eval_folder, f'eval_results_epoch_{epoch+1:03d}.json')
            
            # run validation - 使用羽毛球专用的验证函数
            if isinstance(det_eval, BadmintonDetection):
                eval_results = badminton_valid_one_epoch(
                    val_loader,
                    model_ema.module,
                    epoch,
                    evaluator=det_eval,
                    output_file=eval_file,
                    tb_writer=tb_writer,
                    print_freq=args.print_freq,
                    compute_loss=True  # 计算验证集loss
                )
            else:
                eval_results = valid_one_epoch(
                    val_loader,
                    model_ema.module,
                    epoch,
                    evaluator=det_eval,
                    output_file=eval_file,
                    tb_writer=tb_writer,
                    print_freq=args.print_freq
                )

            print(eval_results['validation_loss'])
            print(eval_results["results"])

            # read and store evaluation results
            if os.path.exists(eval_file):
                with open(eval_file, 'r', encoding='utf-8') as f:
                    eval_results = json.load(f)
                    all_eval_results[f'epoch_{epoch+1:03d}'] = eval_results
                    
                    # log to tensorboard
                    if 'results' in eval_results:
                        for key, value in eval_results['results'].items():
                            if isinstance(value, (int, float)):
                                tb_writer.add_scalar(f'Validation/{key}', value, epoch)
                    
                    # log validation loss to tensorboard
                    if 'validation_loss' in eval_results:
                        for key, value in eval_results['validation_loss'].items():
                            if isinstance(value, (int, float)):
                                tb_writer.add_scalar(f'Validation_Loss/{key}', value, epoch)
                    
                    print(f"Validation results saved to: {eval_file}")

        # save checkpoint
        if (
            ((epoch + 1) == max_epochs) or
            ((args.ckpt_freq > 0) and ((epoch + 1) % args.ckpt_freq == 0))
        ):
            save_states = {
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'scheduler': scheduler.state_dict(),
                'optimizer': optimizer.state_dict(),
            }

            save_states['state_dict_ema'] = model_ema.module.state_dict()
            save_checkpoint(
                save_states,
                False,
                file_folder=ckpt_folder,
                file_name='epoch_{:03d}.pth.tar'.format(epoch + 1)
            )

    # save all evaluation results summary
    summary_file = os.path.join(eval_folder, 'all_eval_results.json')
    with open(summary_file, 'w', encoding='utf-8') as f:
        json.dump(all_eval_results, f, indent=2, ensure_ascii=False)
    print(f"\nAll evaluation results saved to: {summary_file}")
    
    # find best epoch
    best_epoch = None
    best_map = 0.0
    for epoch_key, results in all_eval_results.items():
        if 'results' in results and 'mAP' in results['results']:
            current_map = results['results']['mAP']
            if current_map > best_map:
                best_map = current_map
                best_epoch = epoch_key
    
    if best_epoch:
        print(f"Best epoch: {best_epoch} with mAP: {best_map:.4f}")
        # save best epoch info
        best_info = {
            'best_epoch': best_epoch,
            'best_mAP': best_map,
            'best_results': all_eval_results[best_epoch]
        }
        with open(os.path.join(eval_folder, 'best_results.json'), 'w', encoding='utf-8') as f:
            json.dump(best_info, f, indent=2, ensure_ascii=False)

    # 绘制训练曲线
    config_name = os.path.basename(cfg_filename) if cfg_filename else "unknown_config"
    plot_training_curves(all_eval_results, ckpt_folder, config_name)

    # wrap up
    tb_writer.close()
    print("All done!")
    return

################################################################################
if __name__ == '__main__':
    """Entry Point"""
    # the arg parser
    parser = argparse.ArgumentParser(
      description='Train a point-based transformer for action localization with validation')
    parser.add_argument('config', metavar='DIR',
                        help='path to a config file')
    parser.add_argument('-p', '--print-freq', default=10, type=int,
                        help='print frequency (default: 10 iterations)')
    parser.add_argument('-c', '--ckpt-freq', default=5, type=int,
                        help='checkpoint frequency (default: every 5 epochs)')
    parser.add_argument('-e', '--eval-freq', default=5, type=int,
                        help='evaluation frequency (default: every 5 epochs)')
    parser.add_argument('--output', default='', type=str,
                        help='name of exp folder (default: none)')
    parser.add_argument('--resume', default='', type=str, metavar='PATH',
                        help='path to a checkpoint (default: none)')
    args = parser.parse_args()
    main(args)
