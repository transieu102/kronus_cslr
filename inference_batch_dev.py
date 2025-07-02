import torch
import pickle
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd
from strhub.data.utils import build_tokenizer_from_csv
from dev_data_loader import PoseDataset
import os
from torch.utils.data import DataLoader
import datetime
from model_hub import MODEL_SYSTEMS
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)
def cslr_collate_fn(batch):
    sample_id, poses, label_list = zip(*batch)
    max_len = max(pose.shape[0] for pose in poses)
    padded_poses = []
    for pose in poses:
        pad_size = max_len - pose.shape[0]
        if pad_size > 0:
            pad = torch.zeros((pad_size, pose.shape[1], pose.shape[2]), dtype=pose.dtype)
            pose = torch.cat([pose, pad], dim=0)
        padded_poses.append(pose)
    poses_tensor = torch.stack(padded_poses)
    return list(sample_id), poses_tensor, list(label_list)
def main():
    pose_pkl = "data/pose_data_isharah1000_hands_lips_body_May12.pkl"
    test_csv = "data/si/dev.csv"  # file chỉ có cột id

    checkpoint_path = "/home/sieut/kronus/logs/MultiRegionCSLRSystemVersion3/runs_20250625_174218/version_0/checkpoints/last.ckpt"
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "config.yaml")
    predict_folder = 'predictions'
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(predict_folder,f"test_{timestamp}"), exist_ok=True)
    predictions_file = os.path.join(predict_folder,f"test_{timestamp}", "dev.csv")
    model_path_file_txt = os.path.join(predict_folder,f"test_{timestamp}", "info.txt")
    config = load_config(config_path)
    batch_size = config["trainer"]["batch_size"]
    with open(model_path_file_txt, "w") as f:
        f.write(f"checkpoint_path: {checkpoint_path}\n")
        f.write(f"config_path: {config_path}\n")
        f.write(f"pose_pkl: {pose_pkl}\n")
        f.write(f"test_csv: {test_csv}\n")
        f.write(f"predict_folder: {predict_folder}\n")
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"predictions_file: {predictions_file}\n")
        f.write(f"batch_size: {batch_size}\n")
        # f.write('ctcdecode: BeamSearch\n')

    # Build tokenizer từ train+val
    # Build tokenizer từ train+val csv
    if config["dataset"]["tokenizer_type"] == "both":
        tokenizer_ctc = build_tokenizer_from_csv(
            [config["dataset"]["train_csv"], config["dataset"]["val_csv"]],
            tokenizer_type="ctc"
        )
        tokenizer_entropy = build_tokenizer_from_csv(
            [config["dataset"]["train_csv"], config["dataset"]["val_csv"]],
            tokenizer_type="seq2seq"
        )
    else:
        tokenizer = build_tokenizer_from_csv(
            [config["dataset"]["train_csv"], config["dataset"]["val_csv"]],
            tokenizer_type=config["dataset"]["tokenizer_type"]
        )


    # Đọc danh sách id từ test_csv (giữ nguyên thứ tự)
    test_ids = pd.read_csv(test_csv, delimiter=",")["id"].astype(str).tolist()

    # Load pose_dict trực tiếp để truy xuất nhanh
    with open(pose_pkl, "rb") as f:
        pose_dict = pickle.load(f)

    # Tạo một instance PoseDataset để dùng lại các hàm xử lý pose
    target_enc_df = pd.DataFrame()
    dummy_dataset = PoseDataset(
        dataset_name2="test",
        pkl_path=pose_pkl,
        label_csv=test_csv,
        split_type="test",
        target_enc_df=target_enc_df,
        transform=None,
        augmentations=False,
        augmentations_prob=0.0,
        additional_joints=config["dataset"].get("additional_joints", True)
    )
    # Model system
    # Model system
    # MODEL_SYSTEMS = {
    #     "CSLRTransformerBaseline": CSLRBaselineSystem,
    #     "CSLRTransformerBaselineIndependent": CSLRBaselineSystemIndependent,
    #     "GCNTransformerBaseline": GCNBaselineSystem,
    #     "CSLRTransformerAutoregressivePermutate": CSLRAutoregressiveSystemPermutate,
    #     "CSLRTransformerAutoregressive": CSLRAutoregressiveSystem,
    #     "GNN_AR": GNN_AR_System,
    #     "BaselineAttentionLinear": BaselineAttentionLinear,
    #     "CSLRPositionSpecificSystem": CSLRPositionSpecificSystem
    # }

    system_cls = MODEL_SYSTEMS[config["model"]["name"]]
    if config["dataset"]["tokenizer_type"] == "both":
            system = system_cls.load_from_checkpoint(
                checkpoint_path,
                tokenizer_ctc=tokenizer_ctc,
                tokenizer_entropy=tokenizer_entropy,
                config=config
            )
    else:
            system = system_cls.load_from_checkpoint(
                checkpoint_path,
                tokenizer=tokenizer,
                config=config
            )

    # Load model
    # system = CSLRBaselineSystem.load_from_checkpoint(
    #     checkpoint_path,
    #     tokenizer=tokenizer,
    #     batch_size=1,
    #     lr=config["trainer"]["lr"],
    #     warmup_pct=config["trainer"]["warmup_pct"],
    #     weight_decay=config["trainer"]["weight_decay"],
    #     input_dim=config["model"]["input_dim"],
    #     hidden_dim=config["model"]["hidden_dim"],
    #     num_layers=config["model"]["num_layers"],
    #     num_heads=config["model"]["num_heads"],
    #     conv_channels=config["model"]["conv_channels"],
    #     mlp_hidden=config["model"]["mlp_hidden"],
    #     num_classes=len(tokenizer),
    #     dropout=config["model"]["dropout"],
    #     max_input_len=config["model"]["max_input_len"],
    #     max_output_len=config["model"]["max_output_len"]
    # )
    system.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    system = system.to(device)

    #dataloader
    test_loader = DataLoader(
        dummy_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=cslr_collate_fn

    )
    result = {}
    max_t = 0
    with torch.no_grad():
        for batch in tqdm(test_loader):
            sample_id, pose, _ = batch
            max_t = max(max_t, pose.shape[1])
            pose = pose.to(device)
            # logits = system.fusion_forward(pose)
            logits = system(pose)
            probs = logits.softmax(-1)
            # print("Model output shape:", logits.shape)  # N, T, C
            # print("Vocab size:", len(system.tokenizer._itos))  # hoặc tokenizer.vocab
            # print(len(tokenizer_ctc._itos))
            # input()
            preds, _ = system.tokenizer.decode(probs)
            for i in range(len(sample_id)):
                result[sample_id[i]] = preds[i]
    with open(predictions_file, "w") as pred_file:
        pred_file.write("id,gloss\n")
        
        for sample_id in tqdm(test_ids, ncols=100, desc="Testing"):
            gloss = ""
            sample_id = sample_id
            try:
                pred_gloss = result[sample_id]
                gloss = ' '.join(pred_gloss)
                # print(pred_gloss)
                # input()
            except Exception as e:
                print(e)
                gloss = ""
            pred_file.write(f"{sample_id},{gloss}\n")

    print(f"Saved final gloss predictions to: {predictions_file}, max_t: {max_t}")

if __name__ == "__main__":
    main()
