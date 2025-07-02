import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
from .augment import (
    augment_jitter, augment_time_warp, augment_dropout, augment_scale, augment_frame_dropout,
    normalize, normalize_face, normalize_body
)
# import json

class CSLRPoseDataset(Dataset):
    def __init__(self, config, label_csv, pose_pkl, split_type, augment=True, augment_prob=0.5, additional_joints=True, get_id=False):
        """
        Args:
            config: dict cấu hình augment, min/max len, ...
            label_csv: path tới file csv chứa id và gloss/annotation
            pose_pkl: path tới file pickle chứa keypoints
            split_type: 'train', 'dev', 'test'
            augment: bật/tắt augment
            augment_prob: xác suất augment
            additional_joints: có dùng thêm face/body không
        """
        self.config = config
        self.split_type = split_type
        self.augment = augment
        self.augment_prob = augment_prob
        self.additional_joints = additional_joints
        self.min_len = config.get('min_len', 32)
        self.max_len = config.get('max_len', 1000)
        self.get_id = get_id
        with open(pose_pkl, 'rb') as f:
            self.pose_dict = pickle.load(f)

        self.files = []
        self.labels = []
        self.all_data = pd.read_csv(label_csv, delimiter=",")
        self.all_data = self.all_data[self.all_data["id"].notna()]
        self.all_data = self.all_data[self.all_data["gloss"].notna()]
        label_col = "gloss"
        self.max_out_len = 0 #not include end token
        for _, row in self.all_data.iterrows():
            sample_id = str(row["id"])
            if sample_id in self.pose_dict.keys():
                gloss = str(row[label_col])
                gloss_tokens = gloss.split()
                if len(gloss_tokens) > self.max_out_len:
                    self.max_out_len = len(gloss_tokens)
                self.files.append(sample_id)
                self.labels.append(gloss)
        # #sort by length
        # self.files = sorted(self.files, key=lambda x: len(self.labels[self.files.index(x)].split()))
        # self.labels = sorted(self.labels, key=lambda x: len(x.split()))
        print(f"Loaded {len(self.files)} samples for split: {split_type}, max out len: {self.max_out_len}")

    def __len__(self):
        return len(self.files)

    def pad_or_crop_sequence(self, sequence):
        T, J, D = sequence.shape
        if T < self.min_len:
            pad_len = self.min_len - T
            pad = np.zeros((pad_len, J, D))
            sequence = np.concatenate((sequence, pad), axis=0)
        if sequence.shape[0] > self.max_len:
            sequence = sequence[:self.max_len]
        return sequence

    def readPose(self, sample_id):
        pose_data = self.pose_dict[sample_id]['keypoints']
        if pose_data is None or pose_data.shape[0] == 0:
            raise ValueError(f"Error loading pose data for {sample_id}")
        T, J, D = pose_data.shape
        aug = False
        if self.augment and np.random.rand() < self.augment_prob:
            aug = True
            angle = np.radians(np.random.uniform(-13, 13))
            pose_data = augment_time_warp(pose_data)
            pose_data = augment_frame_dropout(pose_data)
        right_hand = pose_data[:, 0:21, :2]
        left_hand = pose_data[:, 21:42, :2]
        NUM_LIPS = 19
        lips = pose_data[:, 42:42+NUM_LIPS, :2]
        body = pose_data[:,42+NUM_LIPS:]
        right_joints, left_joints, face_joints, body_joints = [], [], [], []
        for ii in range(T):
            rh = right_hand[ii]
            lh = left_hand[ii]
            fc = lips[ii]
            bd = body[ii]
            if rh.sum() == 0:
                rh[:] = right_joints[-1] if ii != 0 else np.zeros((21, 2))
            else:
                if aug:
                    rh = augment_jitter(rh)
                    rh = augment_scale(rh)
                    rh = augment_dropout(rh)
                rh = normalize(rh)
            if lh.sum() == 0:
                lh[:] = left_joints[-1] if ii != 0 else np.zeros((21, 2))
            else:
                if aug:
                    lh = augment_jitter(lh)
                    lh = augment_scale(lh)
                    lh = augment_dropout(lh)
                lh = normalize(lh)
            if fc.sum() == 0:
                fc[:] = face_joints[-1] if ii != 0 else np.zeros((len(fc), 2))
            else:
                fc = normalize_face(fc)
            if bd.sum() == 0:
                bd[:] = body_joints[-1] if ii != 0 else np.zeros((len(bd), 2))
            else:
                bd = normalize_body(bd)
            right_joints.append(rh)
            left_joints.append(lh)
            face_joints.append(fc)
            body_joints.append(bd)
        for ljoint_idx in range(len(left_joints) - 2, -1, -1):
            if left_joints[ljoint_idx].sum() == 0:
                left_joints[ljoint_idx] = left_joints[ljoint_idx + 1].copy()
        for rjoint_idx in range(len(right_joints) - 2, -1, -1):
            if right_joints[rjoint_idx].sum() == 0:
                right_joints[rjoint_idx] = right_joints[rjoint_idx + 1].copy()
        concatenated_joints = np.concatenate((right_joints, left_joints), axis=1)
        if self.additional_joints:
            concatenated_joints = np.concatenate((concatenated_joints, face_joints, body_joints), axis=1)
        return concatenated_joints

    def __getitem__(self, idx):
        sample_id = self.files[idx]
        pose = self.readPose(sample_id)
        pose = self.pad_or_crop_sequence(pose)
        pose_len = pose.shape[0]
        pose = torch.from_numpy(pose).float()
        # label_tokens = self.labels[idx]
        label = self.labels[idx]
        if self.get_id:
            return sample_id, pose, label, pose_len
        return pose, label

class CSLRRandomSegmentDataset(CSLRPoseDataset):
    def __init__(self, config, label_csv, pose_pkl, segment_csv, split_type, 
                 augment=True, augment_prob=0.5, additional_joints=True, get_id=False,
                 random_segment_prob=0.3, random_label_prob=0.2, min_random_len=2, max_random_len=5):
        """
        Args:
            config: dict cấu hình augment, min/max len, ...
            label_csv: path tới file csv chứa id và gloss/annotation
            pose_pkl: path tới file pickle chứa keypoints
            segment_csv: path tới file csv chứa thông tin segment (video_id, gloss, start_frame, end_frame, duration)
            split_type: 'train', 'dev', 'test'
            augment: bật/tắt augment
            augment_prob: xác suất augment
            additional_joints: có dùng thêm face/body không
            random_segment_prob: xác suất augment bằng cách ghép segment ngẫu nhiên
            random_label_prob: xác suất tạo label ngẫu nhiên
            min_random_len: độ dài tối thiểu của label ngẫu nhiên
            max_random_len: độ dài tối đa của label ngẫu nhiên
        """
        super().__init__(config, label_csv, pose_pkl, split_type, augment, augment_prob, additional_joints, get_id)
        
        # Load segment information
        self.segment_df = pd.read_csv(segment_csv)
        self.synth_gloss_df = pd.read_csv(config["synth_gloss_csv"]).dropna()
        print(self.synth_gloss_df.head())
        self.synthentic_gloss_dict = {}
        for _, row in self.synth_gloss_df.iterrows():
            # gloss = json.loads(gloss)
            video_id, gloss = row['video_id'], row['gloss']
            if video_id not in self.synthentic_gloss_dict:
                self.synthentic_gloss_dict[video_id] = []
            self.synthentic_gloss_dict[video_id].append(gloss)
        # self.random_segment_prob = random_segment_prob
        # self.random_label_prob = random_label_prob
        # self.min_random_len = min_random_len
        # self.max_random_len = max_random_len
        self.use_synth_gloss = config.get("use_synth_gloss", False)
        self.use_synth_gloss_prob = config.get("use_synth_gloss_prob", 0.3)
        # Create gloss to segments mapping for faster lookup
        self.gloss_to_segments = {}
        self.video_id_to_gloss = {}
        for _, row in self.segment_df.iterrows():
            gloss = row['gloss']
            start_frame = row['start_frame']
            end_frame = row['end_frame']
            video_id = row['video_id']
            if video_id not in self.video_id_to_gloss:
                self.video_id_to_gloss[video_id] = []
            self.video_id_to_gloss[video_id].append({
                'gloss': gloss,
                'start_frame': start_frame,
                'end_frame': end_frame
            })
            if gloss not in self.gloss_to_segments:
                self.gloss_to_segments[gloss] = []
            self.gloss_to_segments[gloss].append({
                'video_id': row['video_id'],
                'start_frame': row['start_frame'],
                'end_frame': row['end_frame']
            })

    def get_pose_from_segment(self, video_id, start_frame, end_frame):
        """Lấy pose data từ một segment cụ thể và xử lý qua readPose"""
        if video_id not in self.pose_dict:
            return None
        
        # Lưu lại thông tin gốc
        original_pose_data = self.pose_dict[video_id]['keypoints']
        if original_pose_data is None or original_pose_data.shape[0] == 0:
            return None
            
        # Đảm bảo frame indices nằm trong range
        start_frame = max(0, min(start_frame, original_pose_data.shape[0] - 1))
        end_frame = max(0, min(end_frame, original_pose_data.shape[0] - 1))
        
        # Tạo bản sao của pose_dict để tránh ảnh hưởng đến dữ liệu gốc
        temp_pose_dict = {video_id: {'keypoints': original_pose_data[start_frame:end_frame + 1]}}
        
        # Lưu lại pose_dict gốc
        original_pose_dict = self.pose_dict
        try:
            # Thay thế pose_dict tạm thời
            self.pose_dict = temp_pose_dict
            # Sử dụng readPose để xử lý pose data
            processed_pose = self.readPose(video_id)
            return processed_pose
        finally:
            # Khôi phục pose_dict gốc
            self.pose_dict = original_pose_dict

    def concatenate_pose_sequences(self, pose_sequences):
        """Ghép các chuỗi pose đã được xử lý lại với nhau"""
        if not pose_sequences:
            return None
            
        # Lọc bỏ các sequence None
        valid_sequences = [seq for seq in pose_sequences if seq is not None]
        if not valid_sequences:
            return None
            
        return np.concatenate(valid_sequences, axis=0)

    def get_random_segments_for_gloss(self, gloss, num_segments=1):
        """Lấy ngẫu nhiên các segment cho một gloss cụ thể"""
        if gloss not in self.gloss_to_segments:
            return []
            
        segments = self.gloss_to_segments[gloss]
        if not segments:
            return []
            
        # Chọn ngẫu nhiên num_segments segment
        selected_indices = np.random.randint(0, len(segments))
        info = segments[selected_indices]
        start_frame = info['start_frame']
        end_frame = info['end_frame']
        original_pose = self.readPose(info['video_id'])
        original_pose = self.pad_or_crop_sequence(original_pose)
        return original_pose[start_frame:end_frame+1]
        # return start_frame, end_frame

    def create_random_label(self,video_id):
        """Tạo một label ngẫu nhiên"""
        # Chọn độ dài ngẫu nhiên
        # label_len = np.random.randint(self.min_random_len, self.max_random_len + 1)
        if video_id not in self.synthentic_gloss_dict:
            return None
        label_index = np.random.randint(0, len(self.synthentic_gloss_dict[video_id]))
        label = self.synthentic_gloss_dict[video_id][label_index]
        return label
        # Lấy danh sách tất cả các gloss có sẵn
        # available_glosses = list(self.gloss_to_segments.keys())
        # if not available_glosses:
        #     return None
            
        # # Chọn ngẫu nhiên các gloss
        # # selected_glosses = np.random.choice(available_glosses, label_len, replace=True)
        # return ' '.join(selected_glosses)

    def __getitem__(self, idx):
        video_id = self.files[idx]
        original_label = self.labels[idx]
        original_pose = self.readPose(video_id)
        original_pose = self.pad_or_crop_sequence(original_pose)
        original_pose_len = original_pose.shape[0]
        # original_pose = torch.from_numpy(original_pose).float()
        # label_tokens = self.labels[idx]
        # label = self.labels[idx]
        # if self.get_id:
        #     return sample_id, pose, label, pose_len
        # return pose, label
        # Xác định loại augment sẽ sử dụng
        # augment_type = None
        if self.use_synth_gloss and np.random.random() < self.use_synth_gloss_prob:
            # Tạo label ngẫu nhiên và lấy các segment tương ứng
            new_label = self.create_random_label(video_id)
            if new_label is None:
                original_pose = torch.from_numpy(original_pose).float()
                return original_pose, original_label
                # return super().__getitem__(idx)
            # if video_id is None:
            #     return super().__getitem__(idx)
            # try:
            if len(original_label.split()) != len(new_label.split()):
                # print(f"Difference length: {len(original_label.split()) - len(new_label.split())}")
                original_pose = torch.from_numpy(original_pose).float()
                return original_pose, original_label
                # return super().__getitem__(idx)
            # except:
            #     print(original_label)
            #     print(new_label)
            #     input()
        
            index_replace_gloss = None
            for index, gloss in enumerate(original_label.split()):
                if original_label.split()[index] != new_label.split()[index]:
                    # index_replace_gloss.append(index)
                    index_replace_gloss = index
                    break
            if index_replace_gloss is None:
                original_pose = torch.from_numpy(original_pose).float()
                return original_pose, original_label
            #     return super().__getitem__(idx)
            if index_replace_gloss >= len(self.video_id_to_gloss[video_id]):
                original_pose = torch.from_numpy(original_pose).float()
                return original_pose, original_label
            gloss_info = self.video_id_to_gloss[video_id][index_replace_gloss]
            start_replacement = gloss_info['start_frame']
            end_replacement = gloss_info['end_frame']

            new_segments = self.get_random_segments_for_gloss(gloss_info['gloss'])
            # original_pose[start_replacement:end_replacement+1] = new_segments
            original_pose = np.concatenate((original_pose[:start_replacement], new_segments, original_pose[end_replacement+1:]), axis=0)
            # new_pose_len = original_pose.shape[0]
            original_pose = torch.from_numpy(original_pose).float()
            # print(original_label)
            # print(new_label)
            # input()
            return original_pose, new_label
        else:
            # Sử dụng phương thức gốc nếu không augment
            return super().__getitem__(idx)

    