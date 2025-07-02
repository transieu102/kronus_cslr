import yaml
import torch
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer, seed_everything
from pytorch_lightning.callbacks import ModelCheckpoint
from strhub.data.dataset import CSLRPoseDataset
from strhub.data.dataset_segment_augment import CSLRRandomSegmentDataset
from strhub.data.dataset_rgb import CSLRPoseDatasetRGB
from strhub.data.utils import build_tokenizer_from_csv
from pytorch_lightning.loggers import TensorBoardLogger
import os
from model_hub import MODEL_SYSTEMS
import argparse

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
def cslr_collate_fn_rgb(batch):
    poses, rgb_frames, label_list = zip(*batch)
    # print(poses[0].shape)
    # print(rgb_frames[0].shape)
    # input()
    max_len = max(pose.shape[0] for pose in poses)
    padded_poses = []
    padded_rgb_frames = []
    for pose in poses:
        pad_size = max_len - pose.shape[0]
        if pad_size > 0:
            pad = torch.zeros((pad_size, pose.shape[1], pose.shape[2]), dtype=pose.dtype)
            pose = torch.cat([pose, pad], dim=0)
        padded_poses.append(pose)
    poses_tensor = torch.stack(padded_poses)
    max_len_rgb = max(rgb_frame.shape[0] for rgb_frame in rgb_frames)
    for rgb_frame in rgb_frames:
        pad_size = max_len_rgb - rgb_frame.shape[0]
        if pad_size > 0:
            pad = torch.zeros((pad_size, rgb_frame.shape[1], rgb_frame.shape[2], rgb_frame.shape[3]), dtype=rgb_frame.dtype)
            rgb_frame = torch.cat([rgb_frame, pad], dim=0)
        padded_rgb_frames.append(rgb_frame)
    rgb_frames_tensor = torch.stack(padded_rgb_frames)
    return poses_tensor, rgb_frames_tensor, list(label_list)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default="./configs/region_gnn_ctc2stage.yaml", help='Path to config file')
    args = parser.parse_args()
    config = load_config(args.config)
    if config["trainer"].get("seed", None) is not None:
        seed_everything(config["trainer"]["seed"])
        print(f"Seed: {config['trainer']['seed']}")
    else:
        print("No seed provided")
    task = 'us' if '/us/' in config["dataset"]["train_csv"] else 'is'
    if task == 'us':
        config['logging']['log_dir'] += 'us/'
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

    # Dataset & DataLoader
    if config["dataset"].get('use_synth_gloss', False) != False:
        train_dataset = CSLRRandomSegmentDataset(
            config=config["dataset"],
            label_csv=config["dataset"]["train_csv"],
            pose_pkl=config["dataset"]["pose_pkl"],
            segment_csv=config["dataset"]["segment_csv"],
            split_type="train",
            augment=config["dataset"]["augment"],
            augment_prob=config["dataset"]["augment_prob"],
            additional_joints=config["dataset"]["additional_joints"]
        )
    elif config["dataset"].get('frame_size', None) is not None:
        train_dataset = CSLRPoseDatasetRGB(
            config=config["dataset"],
            label_csv=config["dataset"]["train_csv"],
            pose_pkl=config["dataset"]["pose_pkl"],
            split_type="train",
            augment=config["dataset"]["augment"],
            augment_prob=config["dataset"]["augment_prob"],
            additional_joints=config["dataset"]["additional_joints"]
        )
    else:
        train_dataset = CSLRPoseDataset(
            config=config["dataset"],
            label_csv=config["dataset"]["train_csv"],
            pose_pkl=config["dataset"]["pose_pkl"],
            split_type="train",
            augment=config["dataset"]["augment"],
            augment_prob=config["dataset"]["augment_prob"],
            additional_joints=config["dataset"]["additional_joints"]
        )
    if config["dataset"].get('frame_size', None) is not None:
        val_dataset = CSLRPoseDatasetRGB(
            config=config["dataset"],
            label_csv=config["dataset"]["val_csv"],
            pose_pkl=config["dataset"]["pose_pkl"],
            split_type="val",
            augment=False,
            additional_joints=config["dataset"]["additional_joints"]
        )
    else:
        val_dataset = CSLRPoseDataset(
        config=config["dataset"],
        label_csv=config["dataset"]["val_csv"],
        pose_pkl=config["dataset"]["pose_pkl"],
        split_type="val",
        augment=False,
        additional_joints=config["dataset"]["additional_joints"]
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["trainer"]["batch_size"],
        shuffle=True,
        num_workers=config["trainer"]["num_workers"],
        collate_fn=cslr_collate_fn_rgb if config["dataset"].get('frame_size', None) is not None else cslr_collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["trainer"]["batch_size"],
        shuffle=False,
        num_workers=config["trainer"]["num_workers"],
        collate_fn=cslr_collate_fn_rgb if config["dataset"].get('frame_size', None) is not None else cslr_collate_fn
    )

    
    system_cls = MODEL_SYSTEMS[config["model"]["name"]]
    pretrain_path = config["trainer"].get("pretrained", None)
    encoder_pretrained = config["trainer"].get("encoder_pretrained", None)
    if pretrain_path:
        print(f"Loading pretrained model from {pretrain_path}")
        system = system_cls.load_from_checkpoint(
                    pretrain_path,
                    tokenizer=tokenizer,
                    config=config
                )
    elif encoder_pretrained:
        pretrain_config = load_config(os.path.join(os.path.dirname(os.path.dirname(encoder_pretrained)), "config.yaml"))
        pretrained_system = MODEL_SYSTEMS[pretrain_config["model"]["name"]]
        pretrained_system = pretrained_system.load_from_checkpoint(
                    encoder_pretrained,
                    tokenizer=tokenizer,
                    config=pretrain_config
                )
        system = system_cls(
            tokenizer=tokenizer,
            config=config,
            # encoder=pretrained_system
        )
        system.model.input_proj = pretrained_system.model.input_proj
        system.model.pos_encoder = pretrained_system.model.pos_encoder
        system.model.transformer = pretrained_system.model.transformer
        system.model.temporal_pooling = pretrained_system.model.temporal_pooling
    else:
        if config["dataset"]["tokenizer_type"] == "both":
            system = system_cls(
                tokenizer_ctc=tokenizer_ctc,
                tokenizer_entropy=tokenizer_entropy,
                config=config
            )
        else:
            system = system_cls(
                tokenizer=tokenizer,
            config=config
        )

    # Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        # dirpath=config["trainer"]["save_dir"],
        save_top_k=2,
        monitor="val_loss",
        mode="min",
        save_last=True,
        filename="{epoch:02d}-val_loss={val_loss:.4f}-val_wer={val_wer:.4f}"
    )
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    # TensorBoard Logger
    tb_logger = TensorBoardLogger(
        save_dir=os.path.join(config["logging"]["log_dir"], config["model"]["name"]),
        name=f"runs_{timestamp}"
    )

    log_dir = tb_logger.log_dir
    os.makedirs(log_dir, exist_ok=True)
    with open(os.path.join(log_dir, "config.yaml"), "w") as f:
        yaml.dump(config, f)

    # Trainer
    trainer = Trainer(
        max_epochs=config["trainer"]["epochs"],
        accelerator=config["trainer"]["device"],
        callbacks=[checkpoint_callback],
        log_every_n_steps=config["trainer"]["log_interval"],
        # log_every_n_epochs=config["trainer"]["log_interval_epoch"],
        default_root_dir=config["logging"]["log_dir"],
        devices=config["trainer"]["devices"],
        logger=tb_logger,
        check_val_every_n_epoch=config["trainer"]["check_val_every_n_epoch"],
        #shuffle every epoch
        # reload_dataloaders_every_n_epochs=1
        
    )

    # Fit
    trainer.fit(system, train_loader, val_loader)
    # trainer.fit(system, val_loader, val_loader)
