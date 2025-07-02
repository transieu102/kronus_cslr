import torch
import torch.nn.functional as F
from strhub.models.base import CrossEntropySystem
from .model import CSLRTransformerBaseline

class CSLRBaselineSystem(CrossEntropySystem):
    def __init__(self, tokenizer, config):
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
        self.model = CSLRTransformerBaseline(
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
        targets = targets[:, 1:]  # Bỏ <bos>
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

class CSLRPositionSpecificSystem(CSLRBaselineSystem):
    def __init__(self, tokenizer, config):
        super().__init__(tokenizer, config)
        self.position_loss_weight = config["model"].get("position_loss_weight", 0.1)
        self.attention_sparsity_weight = config["model"].get("attention_sparsity_weight", 0.01)
        # Padding cho window size
        self.window_padding = config["model"].get("window_padding", 0.2)  # 20% padding cho mỗi bên

    def forward(self, poses, max_length=None):
        # Trong inference, không cần attention maps
        return self.model(self.tokenizer, poses, max_length)

    def compute_position_loss(self, attention_maps, input_length, target_length):
        """
        Compute loss để khuyến khích mỗi expert tập trung vào vị trí của nó
        attention_maps: [B, max_len, T', T'] - attention weights cho mỗi expert
        input_length: [B] - độ dài thực tế của mỗi sequence trong batch
        target_length: [B] - độ dài của target sequence (số tokens)
        """
        B, L, T, _ = attention_maps.shape
        
        # Tính window size cho mỗi sample trong batch
        # Mỗi token sẽ có window size = input_length / target_length + padding
        base_window_size = (input_length.float() / target_length.float()).long()
        padding = (base_window_size * self.window_padding).long()
        window_sizes = base_window_size + 2 * padding  # Padding cho cả hai bên
        
        # Tạo target attention mask cho mỗi sample
        target_masks = []
        for b in range(B):
            # Tính vị trí bắt đầu của mỗi window
            window_starts = torch.arange(L, device=attention_maps.device) * base_window_size[b]
            # Tạo position indices cho sample này
            position_indices = window_starts.view(1, L, 1, 1)
            # Tính target mask với window size tương ứng
            target_mask = torch.abs(
                torch.arange(T, device=attention_maps.device).view(1, 1, T, 1) - position_indices
            ) <= (window_sizes[b] // 2)  # Chia 2 vì window size là tổng độ rộng
            target_mask = target_mask.float()
            # Normalize target mask
            target_mask = target_mask / (target_mask.sum(dim=2, keepdim=True) + 1e-8)
            
            # Tạo mask cho các tokens thực tế
            token_mask = torch.arange(L, device=attention_maps.device) < target_length[b]
            # Chỉ giữ lại attention cho các tokens thực tế
            target_mask = target_mask * token_mask.view(1, L, 1, 1)
            
            target_masks.append(target_mask)
        
        target_masks = torch.stack(target_masks, dim=0)  # [B, 1, L, T, 1]
        
        # Normalize attention weights
        attention_maps = F.softmax(attention_maps, dim=2)  # [B, L, T', T']
        
        # Tạo mask cho các tokens thực tế
        token_masks = torch.arange(L, device=attention_maps.device).view(1, L, 1, 1) < target_length.view(B, 1, 1, 1)
        
        # Tính KL divergence loss
        # Sử dụng ignore_index để bỏ qua các vị trí padding
        kl_loss = F.kl_div(
            attention_maps.log(), 
            target_masks.squeeze(1),  # [B, L, T, 1]
            reduction='none',
            log_target=False
        )
        
        # Áp dụng mask sau khi tính KL divergence
        kl_loss = (kl_loss * token_masks).sum() / (token_masks.sum() + 1e-8)
        
        return kl_loss

    def compute_attention_sparsity_loss(self, attention_maps, target_length):
        """
        Compute loss để khuyến khích attention weights tập trung (sparse)
        attention_maps: [B, max_len, T', T'] - attention weights cho mỗi expert
        target_length: [B] - độ dài của target sequence (số tokens)
        """
        B, L, T, _ = attention_maps.shape
        
        # Normalize attention weights
        attention_maps = F.softmax(attention_maps, dim=2)  # [B, L, T', T']
        
        # Tạo mask cho các tokens thực tế
        token_masks = torch.arange(L, device=attention_maps.device).view(1, L, 1, 1) < target_length.view(B, 1, 1, 1)
        
        # Tính L1 norm và áp dụng mask
        sparsity_loss = (torch.abs(attention_maps) * token_masks).sum() / (token_masks.sum() + 1e-8)
        
        return sparsity_loss

    def forward_logits_loss(self, poses, labels):
        targets = self.tokenizer.encode(labels, self.device)
        targets = targets[:, 1:]  # Bỏ <bos>
        max_len = targets.shape[1] - 1  # exclude <eos>
        # Trong training, cần attention maps để tính position loss
        logits, attention_maps = self.model(self.tokenizer, poses, max_length=max_len, return_attention=True)
        
        # Cross entropy loss
        ce_loss = F.cross_entropy(logits.flatten(end_dim=1), targets.flatten(), ignore_index=self.tokenizer.pad_id)
        loss_numel = (targets != self.tokenizer.pad_id).sum()
        # print(f"ce_loss: {ce_loss}, loss_numel: {loss_numel}")
        # Tính input length cho mỗi sample trong batch
        # Kiểm tra xem frame có phải là padding hay không bằng cách kiểm tra norm của frame
        # Một frame padding sẽ có norm = 0 vì tất cả các giá trị đều là 0
        frame_norms = torch.norm(poses, dim=(2, 3))  # [B, T] - norm của mỗi frame
        input_length = (frame_norms > 0).sum(dim=1)  # [B] - số frame thực tế
        
        # Tính target length (số tokens) - bỏ eos vì không cần tập trung cho eos
        target_length = torch.logical_and(
            targets != self.tokenizer.pad_id,
            targets != self.tokenizer.eos_id
        ).sum(dim=1)  # [B]
        
        # Position-specific loss với token-based window size
        position_loss = self.compute_position_loss(attention_maps, input_length, target_length)
        
        # Attention sparsity loss
        sparsity_loss = self.compute_attention_sparsity_loss(attention_maps, target_length)
        
        # print(f"ce_loss: {ce_loss}, position_loss: {position_loss}, sparsity_loss: {sparsity_loss}")
        # Combine losses
        total_loss = ce_loss + \
                    self.position_loss_weight * position_loss + \
                    self.attention_sparsity_weight * sparsity_loss
        return logits, total_loss, loss_numel

    def training_step(self, batch, batch_idx):
        poses, labels = batch
        logits, loss, loss_numel = self.forward_logits_loss(poses, labels)
        self.log('train_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        # self.log('train_position_loss', loss, on_step=True, on_epoch=True)
        # self.log('train_sparsity_loss', loss, on_step=True, on_epoch=True)
        return loss


