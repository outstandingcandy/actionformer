import os
import json
import numpy as np
import h5py

import torch
from torch.utils.data import Dataset
from torch.nn import functional as F

from .datasets import register_dataset
from .data_utils import truncate_feats

@register_dataset("badminton")
class BadmintonDataset(Dataset):
    def __init__(
        self,
        is_training,     # if in training mode
        split,           # split, a tuple/list allowing concat of subsets
        feat_folder,     # folder for features
        json_file,       # json file for annotations
        feat_stride,     # temporal stride of the feats
        num_frames,      # number of frames for each feat
        default_fps,     # default fps
        downsample_rate, # downsample rate for feats
        max_seq_len,     # maximum sequence length during training
        trunc_thresh,    # threshold for truncate an action segment
        crop_ratio,      # a tuple (e.g., (0.9, 1.0)) for random cropping
        input_dim,       # input feat dim
        num_classes,     # number of action categories
        file_prefix,     # feature file prefix if any
        file_ext,        # feature file extension if any
        force_upsampling # force to upsample to max_seq_len
    ):
        # file path
        assert os.path.exists(feat_folder) and os.path.exists(json_file)
        assert isinstance(split, tuple) or isinstance(split, list)
        assert crop_ratio == None or len(crop_ratio) == 2
        self.feat_folder = feat_folder
        if file_prefix is not None:
            self.file_prefix = file_prefix
        else:
            self.file_prefix = ''
        self.file_ext = file_ext
        self.json_file = json_file

        # split / training mode
        self.split = split
        self.is_training = is_training

        # features meta info
        self.feat_stride = feat_stride
        self.num_frames = num_frames
        self.input_dim = input_dim
        self.default_fps = default_fps
        self.downsample_rate = downsample_rate
        self.max_seq_len = max_seq_len
        self.trunc_thresh = trunc_thresh
        self.num_classes = num_classes
        self.label_dict = None
        self.crop_ratio = crop_ratio

        # load database and select the subset
        dict_db, label_dict = self._load_json_db(self.json_file)
        
        # "empty" split returns all videos
        if 'training' in split:
            self.data_list = dict_db
        elif 'validation' in split:
            # For now, use all data for training. You can split your data later.
            # Take 20% of data for validation
            n_val = max(1, len(dict_db) // 5)
            self.data_list = dict_db[:n_val]
        else:
            self.data_list = dict_db
            
        self.label_dict = label_dict

        # dataset specific attributes
        self.db_attributes = {
            'dataset_name': 'Badminton Action Detection',
            'tiou_thresholds': np.linspace(0.5, 0.95, 10),
            # we will mask out cliff diving
            'empty_label_ids': [],
        }

    def get_attributes(self):
        return self.db_attributes

    def _load_json_db(self, json_file):
        # load database and select the subset
        with open(json_file, 'r') as fid:
            json_data = json.load(fid)

        json_db = []
        label_dict = {}
        
        # Load action mapping
        action_mapping_file = os.path.join(os.path.dirname(json_file), 'action_mapping.json')
        if os.path.exists(action_mapping_file):
            with open(action_mapping_file, 'r') as f:
                mapping_data = json.load(f)
                label_dict = {i: action for i, action in enumerate(mapping_data['actions'])}

        for vid, value in json_data.get('database', json_data).items():
            # Skip if not in the desired split
            if 'subset' in value and value['subset'] not in self.split:
                continue
                
            duration = value['duration']
            
            # get annotations if available
            segments = value.get('segments', [])
            labels = value.get('labels', [])
            
            # process segments and labels
            valid_segs = []
            valid_labels = []
            
            for seg, label in zip(segments, labels):
                # Convert segments to the expected format
                start_time, end_time = seg
                
                # Check segment validity
                if end_time > start_time and end_time <= duration:
                    valid_segs.append([start_time, end_time])
                    valid_labels.append(label)
            
            json_db.append({
                'id': vid,
                'duration': duration,
                'segments': valid_segs,
                'labels': valid_labels
            })

        return json_db, label_dict

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        # directly return a (truncated) data point (so it is very fast!)
        # auto batching will be disabled in the subsequent dataloader
        # instead the model will need to decide how to batch / preprocess the data
        video_item = self.data_list[idx]

        # load features - each video has its own HDF5 file
        feat_file = os.path.join(self.feat_folder, f"{video_item['id']}{self.file_ext}")
        
        # Load features from individual HDF5 file
        if self.file_ext == '.hdf5':
            if os.path.exists(feat_file):
                with h5py.File(feat_file, 'r') as hf:
                    feats = np.array(hf['features'])
            else:
                # Create dummy features if not found
                print(f"Warning: Features not found for {video_item['id']} at {feat_file}")
                feats = np.random.randn(100, self.input_dim).astype(np.float32)
        else:
            feats = np.load(feat_file).astype(np.float32)
        
        # convert to float32
        feats = feats.astype(np.float32)
        
        # deal with downsampling (= increased feat stride)
        feats = feats[::self.downsample_rate, :]
        feat_stride = self.feat_stride * self.downsample_rate
        feat_offset = 0.5 * self.num_frames / feat_stride
        
        # T x C -> C x T
        feats = torch.from_numpy(np.ascontiguousarray(feats.transpose()))

        # convert time stamp (in second) into temporal feature grids
        # ok to have small negative values here
        if video_item['segments'] is not None:
            segments = torch.from_numpy(
                (np.array(video_item['segments']) * self.default_fps / feat_stride - feat_offset)
            )
            labels = torch.from_numpy(np.array(video_item['labels']))
        else:
            segments, labels = None, None

        # return a data dict
        data_dict = {'video_id'        : video_item['id'],
                     'feats'           : feats,      # C x T
                     'segments'        : segments,   # N x 2
                     'labels'          : labels,     # N
                     'fps'             : self.default_fps,
                     'duration'        : video_item['duration'],
                     'feat_stride'     : feat_stride,
                     'feat_num_frames' : self.num_frames}

        # truncate the features during training
        if self.is_training and (segments is not None):
            data_dict = truncate_feats(
                data_dict, self.max_seq_len, self.trunc_thresh, self.crop_ratio
            )

        return data_dict