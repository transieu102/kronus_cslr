import torch
import torch.nn.functional as F
from strhub.models.base import CrossEntropySystem
from .model import CSLRTransformerAutoregressive

class CSLRAutoregressiveSystem(CrossEntropySystem):
    def __init__(self, tokenizer, config):
        # def __init__(self, tokenizer, config):
        batch_size = config["trainer"]["batch_size"]
        lr = config["trainer"]["lr"]
        warmup_pct = config["trainer"]["warmup_pct"]
        weight_decay = config["trainer"]["weight_decay"]
        input_dim = config["model"]["input_dim"]
        hidden_dim = config["model"]["hidden_dim"]
        num_layers = config["model"]["num_layers"]
        num_heads = config["model"]["num_heads"]
        conv_channels = config["model"]["conv_channels"]
        mlp_hidden = config["model"]["mlp_hidden"]
        num_classes = len(tokenizer)
        dropout = config["model"]["dropout"]
        max_input_len = config["model"]["max_input_len"]
        max_output_len = config["model"]["max_output_len"]
        num_decoder_layers = config["model"]["num_decoder_layers"]
        super().__init__(tokenizer, batch_size, lr, warmup_pct, weight_decay)
        self.model = CSLRTransformerAutoregressive(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            num_layers=num_layers,
            num_heads=num_heads,
            conv_channels=conv_channels,
            mlp_hidden=mlp_hidden,
            num_classes=num_classes,
            dropout=dropout,
            max_input_len=max_input_len,
            max_output_len=max_output_len,
            num_decoder_layers=num_decoder_layers
        )

    def forward(self, poses, max_length=None):
        return self.model(self.tokenizer, poses, max_length)

    def forward_logits_loss(self, poses, labels):
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Remove <bos>
        max_len = targets.shape[1] - 1  # exclude <eos>
        logits = self(poses, max_length=max_len)
        loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.tokenizer.pad_id)
        loss_numel = (targets != self.tokenizer.pad_id).sum()
        return logits, loss, loss_numel

    def training_step(self, batch, batch_idx):
        poses, labels = batch
        logits, loss, loss_numel = self.forward_logits_loss(poses, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss 