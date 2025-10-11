"""
适用于羽毛球数据集的评估指标
"""
import os
import json
import numpy as np
import pandas as pd
from .nms import batched_nms
from .metrics import remove_duplicate_annotations

def load_gt_seg_from_badminton_json(json_file, split=None, label='label', label_offset=0):
    """
    从羽毛球数据集的JSON文件加载ground truth segments
    """
    # load json file
    with open(json_file, "r", encoding="utf8") as f:
        json_db = json.load(f)
    
    # 羽毛球数据集没有'database'键，直接是视频数据
    if 'database' in json_db:
        json_db = json_db['database']
    
    vids, starts, stops, labels = [], [], [], []
    for k, v in json_db.items():
        # filter based on split - 羽毛球数据集可能没有subset字段
        if split is not None and 'subset' in v:
            if v['subset'].lower() != split:
                continue
        
        # 处理segments和labels
        if 'segments' in v and 'labels' in v:
            segments = v['segments']
            segment_labels = v['labels']
            
            if segments and segment_labels:
                # video id
                vids += [k] * len(segments)
                # start and end times
                for seg in segments:
                    starts.append(float(seg[0]))
                    stops.append(float(seg[1]))
                # labels
                for lbl in segment_labels:
                    labels.append(int(lbl) + label_offset)

    # convert to numpy arrays
    vids = np.array(vids)
    starts = np.array(starts)
    stops = np.array(stops)
    labels = np.array(labels)
    
    return vids, starts, stops, labels

class BadmintonDetection(object):
    """
    适用于羽毛球数据集的检测评估类
    """
    def __init__(self, ant_file, split=None, tiou_thresholds=np.linspace(0.5, 0.95, 10)):
        self.tiou_thresholds = tiou_thresholds
        self.ap = None
        self.split = split
        
        # 羽毛球动作名称映射
        self.action_names = {
            0: "发球", 1: "非发球"
        }
        
        # 加载ground truth
        self.ground_truth = self._load_ground_truth(ant_file, split)
        
    def _load_ground_truth(self, ant_file, split):
        """加载ground truth数据"""
        vids, starts, stops, labels = load_gt_seg_from_badminton_json(
            ant_file, split=split
        )
        
        gt_base = pd.DataFrame({
            'video-id': vids,
            't-start': starts,
            't-end': stops,
            'label': labels
        })
        
        return gt_base
    
    def _get_predictions_with_label(self, prediction_by_label, label_name, cidx):
        """Get all predicitons of the given label. Return empty DataFrame if there
        is no predcitions with the given label.
        """
        try:
            return prediction_by_label.get_group(cidx).reset_index(drop=True)
        except:
            # 只在verbose模式下显示警告，避免过多输出
            # print('Warning: No predictions of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def _get_ground_truth_with_label(self, ground_truth_by_label, label_name, cidx):
        """Get all ground truth of the given label. Return empty DataFrame if there
        is no ground truth with the given label.
        """
        try:
            return ground_truth_by_label.get_group(cidx).reset_index(drop=True)
        except:
            # 只在verbose模式下显示警告，避免过多输出
            # print('Warning: No ground truth of label \'%s\' were provdied.' % label_name)
            return pd.DataFrame()

    def wrapper_compute_average_precision(self):
        """Computes average precision for each class in the subset.
        """
        ap = np.zeros((len(self.tiou_thresholds), self.activity_index.shape[0]))

        # Adaptation to query faster
        ground_truth_by_label = self.ground_truth.groupby('label')
        prediction_by_label = self.prediction.groupby('label')

        results = {}
        results['label'] = self.activity_index[:, 1].tolist()
        for tidx, tiou in enumerate(self.tiou_thresholds):
            results[tiou] = []
            for cidx, label_name in enumerate(self.activity_index[:, 1]):
                gt_cidx = int(self.activity_index[cidx, 0])
                
                ground_truth_cidx = self._get_ground_truth_with_label(
                    ground_truth_by_label, label_name, gt_cidx)
                prediction_cidx = self._get_predictions_with_label(
                    prediction_by_label, label_name, gt_cidx)
                ap[tidx, cidx] = compute_average_precision_detection(
                    ground_truth_cidx, prediction_cidx, tiou)
                results[tiou].append(ap[tidx, cidx])

        return results, ap

    def evaluate(self, prediction, activity_index=None, verbose=True):
        """Evaluates a prediction file. For the detection task we measure the
        interpolated mean average precision to measure the performance of a
        method.
        """
        self.prediction = prediction
        
        # 如果没有提供activity_index，从ground truth推断
        if activity_index is None:
            unique_labels = sorted(self.ground_truth['label'].unique())
            self.activity_index = np.array([[i, self.action_names.get(i, f'action_{i}')] for i in unique_labels], dtype=object)
        else:
            self.activity_index = activity_index

        if verbose:
            print('[INIT] Loaded annotations from {} subset.'.format(self.split))
            nr_gt = len(self.ground_truth)
            print('\tNumber of ground truth instances: {}'.format(nr_gt))
            nr_pred = len(self.prediction)
            print('\tNumber of predictions: {}'.format(nr_pred))
            print('\tFixed threshold for tiou score: {}'.format(self.tiou_thresholds))

        # Compute average precision.
        self.results, self.ap = self.wrapper_compute_average_precision()

        if verbose:
            self.print_results()

        return self.results

    def print_results(self):
        """Print the results"""
        
        # 简化的输出，只显示关键指标
        print('\n=== 羽毛球动作检测评估结果 ===')
        
        # 计算不同IoU阈值下的mAP
        map_at_iou = {}
        for i, tiou in enumerate([0.5, 0.75, 0.95]):
            if tiou in self.tiou_thresholds:
                idx = np.where(self.tiou_thresholds == tiou)[0][0]
                map_at_iou[f'mAP@{tiou}'] = self.ap[idx].mean()
        
        overall_map = self.ap.mean()
        map_at_iou['mAP'] = overall_map
        
        # 显示整体性能
        print('整体性能:')
        for key, value in map_at_iou.items():
            print(f'  {key}: {value:.4f}')
        
        # 显示各动作类别的平均AP（所有IoU阈值平均）
        print('\n各动作类别性能 (平均AP):')
        class_ap = self.ap.mean(axis=0)
        for cidx, ap in enumerate(class_ap):
            activity_name = self.activity_index[cidx, 1]
            print(f'  {activity_name}: {ap:.4f}')
        
        # 显示详细的IoU阈值结果（可选）
        if len(self.tiou_thresholds) <= 5:  # 只在IoU阈值较少时显示详细结果
            print('\n详细结果:')
            for tiou, tiou_ap in zip(self.tiou_thresholds, self.ap):
                print(f'  tIoU={tiou:.2f}: mAP={tiou_ap.mean():.4f}')
        
        return map_at_iou

def compute_average_precision_detection(ground_truth, prediction, tiou_threshold=0.5):
    """Compute average precision (detection task) between ground truth and
    predictions data frames. If multiple predictions occurs for the same
    predicted segment, only the one with highest score is matches as
    true positive. This code is greatly inspired by Pascal VOC devkit.

    Parameters
    ----------
    ground_truth : df
        Data frame containing the ground truth instances.
        Required fields: ['video-id', 't-start', 't-end']
    prediction : df
        Data frame containing the prediction instances.
        Required fields: ['video-id, 't-start', 't-end', 'score']
    tiou_threshold : float
        Temporal intersection over union threshold.

    Outputs
    -------
    ap : float
        Average precision score.
    """
    ap = 0.
    if prediction.empty:
        return ap

    npos = float(len(ground_truth))
    lock_gt = np.ones((len(ground_truth),)) * -1
    # Sort predictions by decreasing score order.
    sort_idx = prediction['score'].values.argsort()[::-1]
    prediction = prediction.iloc[sort_idx].reset_index(drop=True)

    # Initialize true positive and false positive vectors.
    tp = np.zeros(len(prediction))
    fp = np.zeros(len(prediction))

    # Adaptation to query faster
    ground_truth_gbvn = ground_truth.groupby('video-id')

    # Assigning true positive to truly grount truth instances.
    for idx, this_pred in prediction.iterrows():

        try:
            # Check if there is at least one ground truth in the video associated.
            ground_truth_videoid = ground_truth_gbvn.get_group(this_pred['video-id'])
        except Exception as e:
            fp[idx] = 1
            continue

        this_gt = ground_truth_videoid.reset_index()
        tiou_arr = segment_iou(this_pred[['t-start', 't-end']].values,
                               this_gt[['t-start', 't-end']].values)
        # We would like to retrieve the predictions with highest tiou score.
        tiou_sorted_idx = tiou_arr.argsort()[::-1]
        for jdx in tiou_sorted_idx:
            if tiou_arr[jdx] < tiou_threshold:
                fp[idx] = 1
                break
            if lock_gt[this_gt.loc[jdx]['index']] >= 0:
                continue
            # Assign as true positive after the filters above.
            tp[idx] = 1
            lock_gt[this_gt.loc[jdx]['index']] = idx
            break

        if fp[idx] == 0 and tp[idx] == 0:
            fp[idx] = 1

    tp_cumsum = tp.cumsum()
    fp_cumsum = fp.cumsum()
    recall_cumsum = tp_cumsum / npos

    precision_cumsum = tp_cumsum / (tp_cumsum + fp_cumsum)

    return interpolated_prec_rec(precision_cumsum, recall_cumsum)

def segment_iou(target_segment, candidate_segments):
    """Compute the temporal intersection over union between a
    target segment and all the test segments.

    Parameters
    ----------
    target_segment : 1d array
        Temporal target segment containing [starting, ending] times.
    candidate_segments : 2d array
        Temporal candidate segments containing N x [starting, ending] times.

    Outputs
    -------
    tiou : 1d array
        Temporal intersection over union score of the N's candidate segments.
    """
    tt1 = np.maximum(target_segment[0], candidate_segments[:, 0])
    tt2 = np.minimum(target_segment[1], candidate_segments[:, 1])
    # Intersection including Non-negative overlap score.
    segments_intersection = (tt2 - tt1).clip(0)
    # Segment union.
    segments_union = (candidate_segments[:, 1] - candidate_segments[:, 0]) \
      + (target_segment[1] - target_segment[0]) - segments_intersection
    # Compute overlap as the ratio of the intersection
    # over union of two segments.
    tIoU = segments_intersection.astype(float) / segments_union
    return tIoU

def interpolated_prec_rec(prec, rec):
    """Interpolated AP - VOCdevkit from VOC 2011.
    """
    mprec = np.hstack([[0], prec, [0]])
    mrec = np.hstack([[0], rec, [1]])
    for i in range(len(mprec) - 1)[::-1]:
        mprec[i] = max(mprec[i], mprec[i + 1])
    idx = np.where(mrec[1::] != mrec[0:-1])[0] + 1
    ap = np.sum((mrec[idx] - mrec[idx - 1]) * mprec[idx])
    return ap
