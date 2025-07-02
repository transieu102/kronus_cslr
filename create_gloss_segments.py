import yaml
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from strhub.data.dataset import CSLRPoseDataset
from strhub.data.utils import build_tokenizer_from_csv
import os
import pandas as pd
from tqdm import tqdm
from model_hub import MODEL_SYSTEMS
import numpy as np
from datetime import datetime

# Hardcoded paths
CHECKPOINT_PATHS = {
    'us': "/home/sieut/kronus/logsus/RegionGNNCSLRSystemCTC/runs_20250614_142059/version_0/checkpoints/last.ckpt",
    # 'is': "/home/sieut/kronus/logs/RegionGNNCSLRSystemCTC/runs_20250613_114458/version_0/checkpoints/last.ckpt"
}

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def cslr_collate_fn(batch):
    sample_ids, poses, label_list, pose_len = zip(*batch)
    max_len = max(pose.shape[0] for pose in poses)
    padded_poses = []
    for pose in poses:
        pad_size = max_len - pose.shape[0]
        if pad_size > 0:
            pad = torch.zeros((pad_size, pose.shape[1], pose.shape[2]), dtype=pose.dtype)
            pose = torch.cat([pose, pad], dim=0)
        padded_poses.append(pose)
    poses_tensor = torch.stack(padded_poses)
    return list(sample_ids), poses_tensor, list(label_list), list(pose_len)

def create_gloss_segments(task='us', min_duration=4, output_dir=None):
    """
    Create gloss segments using forced alignment from a trained model.
    Args:
        task: 'us' or 'is' for US or IS dataset
        min_duration: minimum duration (in frames) for a valid segment
        output_dir: directory to save output files (if None, will use default path)
    """
    # Set paths
    checkpoint_path = CHECKPOINT_PATHS[task]
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "config.yaml")
    
    # Set output path
    if output_dir is None:
        output_dir = os.path.join("data", "gloss_segments", task)
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_csv = os.path.join(output_dir, f"gloss_segments_{timestamp}.csv")
    
    # Load config and model
    config = load_config(config_path)
    
    # Build tokenizer
    tokenizer = build_tokenizer_from_csv(
        [config["dataset"]["train_csv"], config["dataset"]["val_csv"]],
        tokenizer_type=config["dataset"]["tokenizer_type"]
    )
    
    # Load model
    system_cls = MODEL_SYSTEMS[config["model"]["name"]]
    system = system_cls.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        config=config
    )
    system.eval()
    system.cuda()
    
    # Create dataset and dataloader
    dataset = CSLRPoseDataset(
        config=config["dataset"],
        label_csv=config["dataset"]["train_csv"],
        pose_pkl=config["dataset"]["pose_pkl"],
        split_type="train",
        augment=False,
        additional_joints=config["dataset"]["additional_joints"],
        get_id=True
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config["trainer"]["num_workers"],
        collate_fn=cslr_collate_fn
    )
    
    # Process each batch
    segments_data = []
    with torch.no_grad():
        for batch_idx, (sample_ids, poses, labels, pose_lens) in enumerate(tqdm(dataloader)):
            # Move to GPU
            poses = poses.cuda()
            
            # Get alignments
            alignments = system.model.align(poses, labels, system.tokenizer)
            
            # Process each sample in batch
            for i, (alignment, label, sample_id, pose_len) in enumerate(zip(alignments, labels, sample_ids, pose_lens)):
                # Get original pose length (before padding)
                orig_len = pose_len
                
                # Get gloss tokens
                gloss_tokens = label.split()
                
                # Debug prints
                # print("\nProcessing sample:", sample_id)
                # print("Original label:", label)
                # print("Number of gloss tokens:", len(gloss_tokens))
                # print("Number of alignments:", len(alignment))
                # print("Alignment:", alignment)
                # print("Gloss tokens:", gloss_tokens)
                # print("Original pose length:", orig_len)
                
                # Create segments
                for (start_frame, end_frame), gloss in zip(alignment, gloss_tokens):
                    # Clip to original length
                    start_frame = min(start_frame, orig_len)
                    end_frame = min(end_frame, orig_len)
                    
                    # print(f"Segment: {gloss} [{start_frame}:{end_frame}] (duration: {end_frame - start_frame})")
                    
                    # Skip if segment is too short or invalid
                    if end_frame <= start_frame or end_frame - start_frame < min_duration:
                        print(f"Skipping segment {gloss} - too short or invalid")
                        continue
                        
                    segments_data.append({
                        'video_id': sample_id,
                        'gloss': gloss,
                        'start_frame': int(start_frame),
                        'end_frame': int(end_frame),
                        'duration': int(end_frame - start_frame)
                    })
                # input()
    # Create DataFrame and save to CSV
    df = pd.DataFrame(segments_data)
    df.to_csv(output_csv, index=False)
    print(f"Saved {len(df)} gloss segments to {output_csv}")
    
    # Print some statistics
    print("\nDataset Statistics:")
    print(f"Total number of segments: {len(df)}")
    print(f"Average segment duration: {df['duration'].mean():.2f} frames")
    print(f"Min segment duration: {df['duration'].min()} frames")
    print(f"Max segment duration: {df['duration'].max()} frames")
    print(f"Number of unique glosses: {df['gloss'].nunique()}")
    print(f"Number of unique videos: {df['video_id'].nunique()}")
    
    # Save statistics to a separate file
    stats_file = os.path.join(output_dir, f"stats_{timestamp}.txt")
    with open(stats_file, 'w') as f:
        f.write(f"Dataset Statistics for {task} dataset:\n")
        f.write(f"Generated on: {timestamp}\n")
        f.write(f"Model checkpoint: {checkpoint_path}\n")
        f.write(f"Total number of segments: {len(df)}\n")
        f.write(f"Average segment duration: {df['duration'].mean():.2f} frames\n")
        f.write(f"Min segment duration: {df['duration'].min()} frames\n")
        f.write(f"Max segment duration: {df['duration'].max()} frames\n")
        f.write(f"Number of unique glosses: {df['gloss'].nunique()}\n")
        f.write(f"Number of unique videos: {df['video_id'].nunique()}\n")
    
    return output_csv, stats_file

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--task', type=str, choices=['us', 'is'], default='us', 
                      help='Dataset task (us or is)')
    parser.add_argument('--min-duration', type=int, default=4,
                      help='Minimum duration (in frames) for a valid segment')
    parser.add_argument('--output-dir', type=str, default=None,
                      help='Directory to save output files (default: data/gloss_segments/{task})')
    args = parser.parse_args()
    
    create_gloss_segments(
        task=args.task,
        min_duration=args.min_duration,
        output_dir=args.output_dir
    ) 