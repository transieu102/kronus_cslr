import numpy as np

def rotate(origin, point, angle):
    """Rotates a point around an origin."""
    ox, oy = origin
    px, py = point
    qx = ox + np.cos(angle) * (px - ox) - np.sin(angle) * (py - oy)
    qy = oy + np.sin(angle) * (px - ox) + np.cos(angle) * (py - oy)
    return qx, qy

def augment_jitter(keypoints, std_dev=0.01):
    """Adds Gaussian noise to keypoints."""
    noise = np.random.normal(loc=0, scale=std_dev, size=keypoints.shape)
    return keypoints + noise

def augment_time_warp(pose_data, max_shift=2):
    """Randomly shifts frames to simulate varying signing speed."""
    T = pose_data.shape[0]
    new_data = np.zeros_like(pose_data)
    for i in range(T):
        shift = np.random.randint(-max_shift, max_shift + 1)
        new_idx = np.clip(i + shift, 0, T - 1)
        new_data[i] = pose_data[new_idx]
    return new_data

def augment_dropout(keypoints, drop_prob=0.1):
    """Randomly drops some keypoints."""
    mask = np.random.rand(*keypoints.shape[:1]) > drop_prob
    keypoints *= mask[:, np.newaxis]
    return keypoints

def augment_scale(keypoints, scale_range=(0.8, 1.2)):
    """Randomly scales keypoints."""
    scale = np.random.uniform(*scale_range)
    return keypoints * [scale, scale]

def augment_frame_dropout(pose_data, drop_prob=0.1):
    """Randomly drops full frames."""
    T = pose_data.shape[0]
    mask = np.random.rand(T) > drop_prob
    return pose_data * mask[:, np.newaxis, np.newaxis]

def normalize(pose):
    pose[:,:] -= pose[0]
    pose[:,:] -= np.min(pose, axis=0)
    max_vals = np.max(pose, axis=0)
    pose[:,:] /= max(max_vals)
    pose[:,:] = pose[:,:] - np.mean(pose[:,:])
    pose[:,:] = pose[:,:] / np.max(np.abs(pose[:,:]))
    pose[:,:] = pose[:,:] * 0.5
    return pose

def normalize_face(pose):
    pose[:,:] -= pose[0]
    pose[:,:] -= np.min(pose, axis=0)
    max_vals = np.max(pose, axis=0)
    pose[:,:] /= max(max_vals)
    pose[:,:] = pose[:,:] - np.mean(pose[:,:])
    pose[:,:] = pose[:,:] / np.max(np.abs(pose[:,:]))
    pose[:,:] = pose[:,:] * 0.5
    return pose

def normalize_body(pose):
    pose[:,:] -= pose[0]
    pose[:,:] -= np.min(pose, axis=0)
    max_vals = np.max(pose, axis=0)
    pose[:,:] /= max(max_vals)
    pose[:,:] = pose[:,:] - np.mean(pose[:,:])
    pose[:,:] = pose[:,:] / np.max(np.abs(pose[:,:]))
    pose[:,:] = pose[:,:] * 0.5
    return pose 