import torch
import pickle
from tqdm import tqdm
import yaml
import numpy as np
import pandas as pd
from strhub.data.utils import build_tokenizer_from_csv
from model_hub import MODEL_SYSTEMS
from test_data_loader import PoseDataset
import os
import datetime
def load_config(config_path):
    with open(config_path, "r") as f:
        return yaml.safe_load(f)

def main():
    pose_pkl = "/home/sieut/kronus/data/pose_data_isharah1000_hands_lips_body_May12.pkl"
    test_csv = "/home/sieut/kronus/data/us/dev.csv"  # file chỉ có cột id

    checkpoint_path = "/home/sieut/kronus/logsus/CSLRPositionSpecificSystem/runs_20250606_081648/version_0/checkpoints/epoch=23-val_loss=val_loss=4.4128-val_wer=val_wer=74.6137.ckpt"
    config_path = os.path.join(os.path.dirname(os.path.dirname(checkpoint_path)), "config.yaml")
    predict_folder = 'predictions'
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    os.makedirs(os.path.join(predict_folder,f"test_{timestamp}"), exist_ok=True)
    predictions_file = os.path.join(predict_folder,f"test_{timestamp}", "test.csv")
    model_path_file_txt = os.path.join(predict_folder,f"test_{timestamp}", "info.txt")
    with open(model_path_file_txt, "w") as f:
        f.write(f"checkpoint_path: {checkpoint_path}\n")
        f.write(f"config_path: {config_path}\n")
        f.write(f"pose_pkl: {pose_pkl}\n")
        f.write(f"test_csv: {test_csv}\n")
        f.write(f"predict_folder: {predict_folder}\n")
        f.write(f"timestamp: {timestamp}\n")
        f.write(f"predictions_file: {predictions_file}\n")
    config = load_config(config_path)

    # Build tokenizer từ train+val
    tokenizer = build_tokenizer_from_csv(
        [config["dataset"]["train_csv"], config["dataset"]["val_csv"]],
        tokenizer_type=config["dataset"]["tokenizer_type"]
    )

    # Đọc danh sách id từ test_csv (giữ nguyên thứ tự)
    test_ids = pd.read_csv(test_csv, delimiter="|")["id"].astype(str).tolist()

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
        additional_joints=config["dataset"].get("additional_joints", False)
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
    max_t = 0
    with open(predictions_file, "w") as pred_file:
        pred_file.write("id,gloss\n")
        with torch.no_grad():
            for sample_id in tqdm(test_ids, ncols=100, desc="Testing"):
                gloss = ""
                sample_id = int(sample_id)
                try:
                    # Lấy pose từ pose_dict, xử lý như trong PoseDataset
                    if sample_id not in pose_dict:
                        raise KeyError(f"{sample_id} not in pose_dict")
                    pose = dummy_dataset.readPose(sample_id)
                    pose = dummy_dataset.pad_or_crop_sequence(pose, min_len=32, max_len=1000)
                    pose = torch.from_numpy(pose).float().unsqueeze(0).to(device)  # (1, T, J, D)
                    if pose.shape[1] > max_t:
                        max_t = pose.shape[1]
                    logits = system(pose)
                    probs = logits.softmax(-1)
                    preds, _ = tokenizer.decode(probs)
                    pred_gloss = preds[0]
                    gloss = ' '.join(pred_gloss)
                    # print(pred_gloss)
                    # input()
                except Exception as e:
                    print(e)
                    gloss = ""
                pred_file.write(f"{sample_id},{gloss}\n")

    print(f"Saved final gloss predictions to: {predictions_file}, max frame: {max_t}")

if __name__ == "__main__":
    main()
