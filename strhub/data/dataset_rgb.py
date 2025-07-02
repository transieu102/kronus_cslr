import os
import numpy as np
import torch
from torch.utils.data import Dataset
import pandas as pd
import pickle
import cv2
import mediapipe as mp
from .augment import (
    augment_jitter, augment_time_warp, augment_dropout, augment_scale, augment_frame_dropout,
    normalize, normalize_face, normalize_body
)

class CSLRPoseDatasetRGB(Dataset):
    def __init__(self, config, label_csv, pose_pkl, split_type, augment=True, augment_prob=0.5, additional_joints=True, get_id=False, get_rgb=True):
        """
        Args:
            config: dict cấu hình augment, min/max len, ...
            label_csv: path tới file csv chứa id và gloss/annotation
            pose_pkl: path tới file pickle chứa keypoints
            split_type: 'train', 'dev', 'test'
            augment: bật/tắt augment
            augment_prob: xác suất augment
            additional_joints: có dùng thêm face/body không
            get_rgb: có trả về RGB frame không
        """
        self.config = config
        self.split_type = split_type
        self.augment = augment
        self.augment_prob = augment_prob
        self.additional_joints = additional_joints
        self.min_len = config.get('min_len', 32)
        self.max_len = config.get('max_len', 1000)
        self.get_id = get_id
        self.get_rgb = True
        self.frame_size = config.get('frame_size', (256, 256))
        
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
        print(f"Loaded {len(self.files)} samples for split: {split_type}, max out len: {self.max_out_len}")
    def __len__(self):
        return len(self.files)
    def visualize_pose_frame(self, keypoints, frame_size=(256, 256)):
        """Visualize pose keypoints on a blank RGB frame"""
        # Create blank frame
        frame = np.ones((frame_size[0], frame_size[1], 3), dtype=np.uint8) * 255
        
        # Get parts
        NUM_LIPS = 19
        rh = keypoints[0:21]
        lh = keypoints[21:42]
        lips = keypoints[42:42+NUM_LIPS]
        body = keypoints[42+NUM_LIPS:]
        
        # Combine all points for normalization
        all_points = np.concatenate([rh, lh, lips, body], axis=0)
        
        # Scale all points together to frame size
        x_min, y_min = all_points.min(axis=0)
        x_max, y_max = all_points.max(axis=0)
        scale_x = frame_size[1] / (x_max - x_min + 1e-6)
        scale_y = frame_size[0] / (y_max - y_min + 1e-6)
        scale = min(scale_x, scale_y)
        
        # Apply the same scaling to all parts
        def scale_points(points):
            points = (points - np.array([x_min, y_min])) * scale
            return points.astype(np.int32)
        
        rh = scale_points(rh)
        lh = scale_points(lh)
        lips = scale_points(lips)
        body = scale_points(body)
        
        # Draw connections
        def draw_connections(points, connections, color):
            for conn in connections:
                start_idx, end_idx = conn
                if start_idx < len(points) and end_idx < len(points):
                    cv2.line(frame, tuple(points[start_idx]), tuple(points[end_idx]), color, 1)
        
        # Draw points
        def draw_points(points, color):
            for point in points:
                cv2.circle(frame, tuple(point), 2, color, -1)
        
        # Draw each component
        draw_connections(rh, mp.solutions.hands.HAND_CONNECTIONS, (0, 0, 255))  # Red
        draw_connections(lh, mp.solutions.hands.HAND_CONNECTIONS, (255, 0, 0))  # Blue
        draw_connections(body, mp.solutions.holistic.POSE_CONNECTIONS, (0, 255, 0))  # Green
        
        draw_points(rh, (0, 0, 255))
        draw_points(lh, (255, 0, 0))
        draw_points(lips, (0, 255, 255))
        draw_points(body, (0, 255, 0))
        
        #save frame
        # cv2.imwrite(f"/home/sieut/kronus/rgb_frames/test.png", frame)
        #reshape to chanel, height, width
        frame = frame.transpose(2, 0, 1)
        #normalize to 0-1
        frame = frame / 255.0
        return frame

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
        rgb_frames = []
        
        for ii in range(T):
            rh = right_hand[ii]
            lh = left_hand[ii]
            fc = lips[ii]
            bd = body[ii]
            orginal_frame = np.concatenate([rh, lh, fc, bd], axis=0)
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
            
            if self.get_rgb and ii % 9 == 0:
                # Create visualization frame for current pose
                rgb_frame = self.visualize_pose_frame(orginal_frame, self.frame_size)
                rgb_frames.append(rgb_frame)
        
        for ljoint_idx in range(len(left_joints) - 2, -1, -1):
            if left_joints[ljoint_idx].sum() == 0:
                left_joints[ljoint_idx] = left_joints[ljoint_idx + 1].copy()
        for rjoint_idx in range(len(right_joints) - 2, -1, -1):
            if right_joints[rjoint_idx].sum() == 0:
                right_joints[rjoint_idx] = right_joints[rjoint_idx + 1].copy()
        concatenated_joints = np.concatenate((right_joints, left_joints), axis=1)
        if self.additional_joints:
            concatenated_joints = np.concatenate((concatenated_joints, face_joints, body_joints), axis=1)
        
        if self.get_rgb:
            return concatenated_joints, np.array(rgb_frames)
        return concatenated_joints

    def __getitem__(self, idx):
        sample_id = self.files[idx]
        # if self.get_rgb:
        pose, rgb_frames = self.readPose(sample_id)
        pose = self.pad_or_crop_sequence(pose)
        pose_len = pose.shape[0]
        pose = torch.from_numpy(pose).float()
        rgb_frames = torch.from_numpy(rgb_frames).float() / 255.0  # Normalize to [0,1]
        label = self.labels[idx]
        if self.get_id:
            return sample_id, pose, rgb_frames, label, pose_len
        return pose, rgb_frames, label
        # else:
        #     pose = self.readPose(sample_id)
        #     pose = self.pad_or_crop_sequence(pose)
        #     pose_len = pose.shape[0]
        #     pose = torch.from_numpy(pose).float()
        #     label = self.labels[idx]
        #     if self.get_id:
        #         return sample_id, pose, label, pose_len
        #     return pose, label

    