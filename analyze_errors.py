import torch
import yaml
import pandas as pd
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader
from strhub.data.utils import build_tokenizer_from_csv
from model_hub import MODEL_SYSTEMS
from strhub.data.dataset import CSLRPoseDataset
import os
import datetime
from collections import defaultdict
# import Levenshtein
from utils.metrics import wer_single_list

def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def cslr_collate_fn(batch):
    poses, label_list = zip(*batch)
    max_len = max(pose.shape[0] for pose in poses)
    padded_poses = []
    for pose in poses:
        pad_size = max_len - pose.shape[0]
        if pad_size > 0:
            pad = torch.zeros((pad_size, pose.shape[1], pose.shape[2]), dtype=pose.dtype)
            pose = torch.cat([pose, pad], dim=0)
        padded_poses.append(pose)
    poses_tensor = torch.stack(padded_poses)
    return poses_tensor, list(label_list)

def analyze_errors(checkpoint_path):
    # Load config from checkpoint directory
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "config.yaml")
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at {config_path}")
    
    config = load_config(config_path)
    
    # Determine if this is US or SI dataset based on config
    is_us = '/us/' in config["dataset"]["train_csv"]
    
    # Set up paths based on dataset type
    if is_us:
        pose_pkl = "/home/sieut/kronus/data/pose_data_isharah1000_hands_lips_body_May12.pkl"
        val_csv = "/home/sieut/kronus/data/us/train.csv"
    else:
        pose_pkl = "/home/sieut/kronus/data/pose_data_isharah1000_hands_lips_body_May12.pkl"
        val_csv = "/home/sieut/kronus/data/si/dev.txt"
    
    print(f"Using dataset: {'US' if is_us else 'SI'}")
    print(f"Pose data: {pose_pkl}")
    print(f"Validation CSV: {val_csv}")
    
    # Build tokenizer
    tokenizer = build_tokenizer_from_csv(
        [config["dataset"]["train_csv"], config["dataset"]["val_csv"]],
        tokenizer_type=config["dataset"]["tokenizer_type"]
    )

    # Load validation dataset
    val_dataset = CSLRPoseDataset(
        config=config["dataset"],
        label_csv=val_csv,
        pose_pkl=pose_pkl,
        split_type="train",
        augment=False,
        additional_joints=config["dataset"].get("additional_joints", False)
    )

    val_loader = DataLoader(
        val_dataset,
        # batch_size=config["trainer"]["batch_size"],
        batch_size=1,
        shuffle=False,
        num_workers=config["trainer"]["num_workers"],
        collate_fn=cslr_collate_fn
    )

    # Load model
    system_cls = MODEL_SYSTEMS[config["model"]["name"]]
    system = system_cls.load_from_checkpoint(
        checkpoint_path,
        tokenizer=tokenizer,
        config=config
    )
    system.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = system.to(device)

    # Create output directory
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f"error_analysis/error_analysis_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)

    # Save run info
    with open(os.path.join(output_dir, 'run_info.txt'), 'w') as f:
        f.write(f"Checkpoint: {checkpoint_path}\n")
        f.write(f"Config: {config_path}\n")
        f.write(f"Pose data: {pose_pkl}\n")
        f.write(f"Validation CSV: {val_csv}\n")
        f.write(f"Dataset type: {'US' if is_us else 'SI'}\n")

    # Initialize error tracking
    error_stats = defaultdict(int)
    error_examples = []
    total_samples = 0
    total_wer = 0

    # Run inference and collect errors
    with torch.no_grad():
        for batch in tqdm(val_loader, desc="Analyzing errors"):
            poses, labels = batch
            poses = poses.to(device)
            # print(labels)
            # input()
            # Get predictions
            logits, _, ca_weights = system(poses, len(labels[0].split()))
            print(ca_weights.shape)
            # input()
            probs = logits.softmax(-1)
            preds, _ = tokenizer.decode(probs)
            
            # Compare with ground truth
            for pred, true_label in zip(preds, labels):
                pred_text = ' '.join(pred)
                # true_text = ' '.join(true_label)
                true_text = true_label
                # Calculate WER
                # print(true_text.split(), pred_text.split())
                # input()
                wer = wer_single_list(true_text.split(), pred_text.split())['wer']
                total_wer += wer
                total_samples += 1
                
                # if pred_text != true_text:
                    # Record error example
                error_examples.append({
                    'prediction': pred_text,
                    'ground_truth': true_text,
                    'wer': wer
                })
                print(error_examples[-1])
                input()
                    
                    # # Analyze error patterns
                    # pred_words = set(pred_text.split())
                    # true_words = set(true_text.split())
                    
                    # # Count substitution errors
                    # for word in pred_words - true_words:
                    #     error_stats[f'substitution_{word}'] += 1
                    
                    # # Count deletion errors
                    # for word in true_words - pred_words:
                    #     error_stats[f'deletion_{word}'] += 1

    # Calculate overall statistics
    avg_wer = total_wer / total_samples if total_samples > 0 else 0
    
    # Save error analysis report
    with open(os.path.join(output_dir, 'error_analysis.txt'), 'w') as f:
        f.write(f"Error Analysis Report\n")
        f.write(f"===================\n\n")
        f.write(f"Total samples analyzed: {total_samples}\n")
        f.write(f"Average WER: {avg_wer:.4f}\n\n")
        
        # f.write("Most Common Errors:\n")
        # f.write("-----------------\n")
        # for error_type, count in sorted(error_stats.items(), key=lambda x: x[1], reverse=True)[:20]:
        #     f.write(f"{error_type}: {count}\n")
        
        f.write("\nError Examples:\n")
        f.write("--------------\n")
        for i, example in enumerate(error_examples[:50], 1):
            f.write(f"\nExample {i}:\n")
            f.write(f"Prediction: {example['prediction']}\n")
            f.write(f"Ground Truth: {example['ground_truth']}\n")
            f.write(f"WER: {example['wer']:.4f}\n")

    print(f"Error analysis complete. Results saved to {output_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Analyze prediction errors')
    parser.add_argument('--checkpoint', type=str, required=True, help='Path to model checkpoint')
    
    args = parser.parse_args()
    analyze_errors(args.checkpoint) 