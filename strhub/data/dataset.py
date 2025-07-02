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

    