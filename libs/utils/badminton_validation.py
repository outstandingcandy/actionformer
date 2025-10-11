"""
羽毛球数据集的验证函数
"""
import time
import torch
import pandas as pd
from .train_utils import AverageMeter

def organize_predictions_by_video(predictions_df, evaluator):
    """
    按视频组织预测结果，并添加动作名称
    """
    detailed_predictions = {}
    
    # 获取动作名称映射
    action_names = getattr(evaluator, 'action_names', {})
    
    # 按视频ID分组
    for video_id in predictions_df['video-id'].unique():
        video_preds = predictions_df[predictions_df['video-id'] == video_id]
        
        # 按分数排序
        video_preds = video_preds.sort_values('score', ascending=False)
        
        predictions_list = []
        for _, pred in video_preds.iterrows():
            pred_info = {
                'start_time': float(pred['t-start']),
                'end_time': float(pred['t-end']),
                'label_id': int(pred['label']),
                'label_name': action_names.get(int(pred['label']), f'action_{int(pred["label"])}'),
                'confidence': float(pred['score']),
                'duration': float(pred['t-end'] - pred['t-start'])
            }
            predictions_list.append(pred_info)
        
        detailed_predictions[video_id] = {
            'num_predictions': len(predictions_list),
            'predictions': predictions_list
        }
    
    return detailed_predictions

def badminton_valid_one_epoch(
    val_loader,
    model,
    curr_epoch,
    evaluator = None,
    output_file = None,
    tb_writer = None,
    print_freq = 20,
    compute_loss = True
):
    """Test the model on the validation set for badminton dataset"""
    # either evaluate the results or save the results
    assert (evaluator is not None) or (output_file is not None)

    # set up meters
    batch_time = AverageMeter()
    losses_tracker = {} if compute_loss else None
    
    # switch to evaluate mode
    model.eval()
    # dict for results (for our evaluation code)
    results = {
        'video-id': [],
        't-start' : [],
        't-end': [],
        'label': [],
        'score': []
    }

    # loop over validation set
    start = time.time()
    for iter_idx, video_list in enumerate(val_loader, 0):
        # forward the model (wo. grad)
        with torch.no_grad():
            # 如果需要计算loss，临时设置为训练模式
            if compute_loss:
                model.train()
                losses = model(video_list)
                model.eval()
                
                # 追踪loss
                for key, value in losses.items():
                    if key not in losses_tracker:
                        losses_tracker[key] = AverageMeter()
                    losses_tracker[key].update(value.item())
                
                # 重新进行推理以获取预测结果
                output = model(video_list)
            else:
                output = model(video_list)

            # unpack the results into ANet format
            num_vids = len(output)
            for vid_idx in range(num_vids):
                if output[vid_idx]['segments'].shape[0] > 0:
                    results['video-id'].extend(
                        [output[vid_idx]['video_id']] *
                        output[vid_idx]['segments'].shape[0]
                    )
                    results['t-start'].append(output[vid_idx]['segments'][:, 0])
                    results['t-end'].append(output[vid_idx]['segments'][:, 1])
                    results['label'].append(output[vid_idx]['labels'])
                    results['score'].append(output[vid_idx]['scores'])

        # printing
        if (iter_idx != 0) and iter_idx % (print_freq) == 0:
            # measure elapsed time (sync all kernels)
            torch.cuda.synchronize()
            batch_time.update((time.time() - start) / print_freq)
            start = time.time()

            # print timing
            print('Test: [{0:05d}/{1:05d}]\t'
                  'Time {batch_time.val:.2f} ({batch_time.avg:.2f})'.format(
                  iter_idx, len(val_loader), batch_time=batch_time))

    # gather all stats and evaluate
    results['t-start'] = torch.cat(results['t-start'], dim=0).cpu().numpy()
    results['t-end'] = torch.cat(results['t-end'], dim=0).cpu().numpy()
    results['label'] = torch.cat(results['label'], dim=0).cpu().numpy()
    results['score'] = torch.cat(results['score'], dim=0).cpu().numpy()

    # 打印loss统计（如果计算了）
    if compute_loss and losses_tracker:
        print("\n[Validation Loss]:")
        loss_info = {}
        for key, value in losses_tracker.items():
            print(f"  {key}: {value.avg:.4f}")
            loss_info[key] = float(value.avg)
    
    # call the evaluator
    eval_results = {}
    if evaluator is not None:
        # make pandas dataframe
        df = pd.DataFrame(results)
        # evaluate using the evaluator
        eval_results = evaluator.evaluate(df, verbose=True)
        
        # 转换结果格式以便保存
        formatted_results = {
            'results': {},
            'AP': [],
            'detailed_predictions': {},  # 添加详细预测结果
            'validation_loss': {}  # 添加验证集loss
        }
        
        # 添加loss信息
        if compute_loss and losses_tracker:
            for key, value in losses_tracker.items():
                formatted_results['validation_loss'][key] = float(value.avg)
        
        # 计算各种mAP指标
        if hasattr(evaluator, 'ap') and evaluator.ap is not None:
            # 整体mAP
            formatted_results['results']['mAP'] = float(evaluator.ap.mean())
            
            # 不同IoU阈值的mAP
            for i, tiou in enumerate(evaluator.tiou_thresholds):
                if tiou in [0.5, 0.75, 0.95]:
                    formatted_results['results'][f'mAP@{tiou}'] = float(evaluator.ap[i].mean())
            
            # 每个类别的AP (平均所有IoU阈值)
            class_ap = evaluator.ap.mean(axis=0)
            formatted_results['AP'] = class_ap.tolist()
            formatted_results['results']['AP_per_class'] = class_ap.tolist()
        
        # 保存每个视频的详细预测结果
        formatted_results['detailed_predictions'] = organize_predictions_by_video(df, evaluator)
        
        eval_results = formatted_results

    # dump to a pickle file that can be directly used for evaluation
    if output_file is not None:
        import json
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(eval_results, f, indent=2, ensure_ascii=False)
        print('\n[RESULTS] Action detection results saved to {}'.format(output_file))

    # log mAP to tb_writer
    if tb_writer is not None and eval_results:
        if 'results' in eval_results:
            for key, value in eval_results['results'].items():
                if isinstance(value, (int, float)):
                    tb_writer.add_scalar(f'validation/{key}', value, curr_epoch)

    return eval_results
