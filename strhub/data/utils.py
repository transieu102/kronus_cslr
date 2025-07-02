import re
from abc import ABC, abstractmethod
from itertools import groupby
from typing import Optional, List

import torch
from torch import Tensor
from torch.nn.utils.rnn import pad_sequence
import pandas as pd
# from pyctcdecode import BeamSearchDecoderCTC, Alphabet, build_ctcdecoder
import numpy as np
class BaseGlossTokenizer(ABC):
    """
    Tokenizer cho gloss (word-level) cho bài toán CSLR.
    """
    def __init__(self, vocab_list: List[str], specials_first: tuple = (), specials_last: tuple = ()):  # vocab_list: list các từ
        self._itos = specials_first + tuple(vocab_list) + specials_last
        self._stoi = {s: i for i, s in enumerate(self._itos)}

    def __len__(self):
        return len(self._itos)

    def _tok2ids(self, tokens: List[str]) -> List[int]:
        return [self._stoi[s] for s in tokens.split()]

    def _ids2tok(self, token_ids: List[int], join: bool = True) -> List[str]:
        tokens = [self._itos[i] for i in token_ids]
        return tokens if not join else ' '.join(tokens)

    @abstractmethod
    def encode(self, labels: List[List[str]], device: Optional[torch.device] = None) -> Tensor:
        """
        Encode một batch gloss (danh sách từ) thành tensor cho model.
        Args:
            labels: List các gloss, mỗi gloss là list các từ.
            device: device để tạo tensor.
        Returns:
            Tensor đã pad, shape: N, L
        """
        raise NotImplementedError

    @abstractmethod
    def _filter(self, probs: Tensor, ids: Tensor) -> tuple[Tensor, List[int]]:
        """
        Lọc các token đặc biệt khi decode.
        """
        raise NotImplementedError

    def decode(self, token_dists: Tensor, raw: bool = False) -> tuple[List[List[str]], List[Tensor]]:
        """
        Decode một batch xác suất token thành gloss.
        Args:
            token_dists: softmax probs, shape N, L, C
            raw: trả về gloss chưa lọc đặc biệt
        Returns:
            list gloss (danh sách từ) và list xác suất
        """
        batch_tokens = []
        batch_probs = []
        for dist in token_dists:
            probs, ids = dist.max(-1)
            # print(ids)
            if not raw:
                probs, ids = self._filter(probs, ids)
            # tokens = self._ids2tok(ids.split().tolist(), join=False)
            tokens = self._ids2tok(ids, join=False)
            batch_tokens.append(tokens)
            batch_probs.append(probs)
        return batch_tokens, batch_probs

class GlossTokenizer(BaseGlossTokenizer):
    BOS = '[BOS]'
    EOS = '[EOS]'
    PAD = '[PAD]'

    def __init__(self, vocab_list: List[str]):
        specials_first = (self.EOS,)
        specials_last = (self.BOS, self.PAD)
        super().__init__(vocab_list, specials_first, specials_last)
        self.eos_id, self.bos_id, self.pad_id = [self._stoi[s] for s in specials_first + specials_last]

    def encode(self, labels: List[List[str]], device: Optional[torch.device] = None) -> Tensor:
        batch = [
            torch.as_tensor([self.bos_id] + self._tok2ids(y) + [self.eos_id], dtype=torch.long, device=device)
            for y in labels
        ]
        return pad_sequence(batch, batch_first=True, padding_value=self.pad_id)

    def _filter(self, probs: Tensor, ids: Tensor) -> tuple[Tensor, List[int]]:
        ids = ids.tolist()
        try:
            eos_idx = ids.index(self.eos_id)
        except ValueError:
            eos_idx = len(ids)
        ids = ids[:eos_idx]
        probs = probs[:eos_idx+1]  # include EOS
        return probs, ids

class CTCGlossTokenizer(BaseGlossTokenizer):
    BLANK = 'BLANK'

    def __init__(self, vocab_list: List[str]):
        super().__init__(vocab_list, specials_first=(self.BLANK,))
        self.blank_id = self._stoi[self.BLANK]
        # self.decoder = build_ctcdecoder(list(self._itos))
    def encode(self, labels: List[List[str]], device: Optional[torch.device] = None) -> Tensor:
        batch = [torch.as_tensor(self._tok2ids(y), dtype=torch.long, device=device) for y in labels]
        # target_lengths = torch.tensor([len(seq) for seq in batch], dtype=torch.long, device=device)
        target_lengths = torch.as_tensor(list(map(len, [label.split() for label in labels])), dtype=torch.long, device=device)
        # flattened_targets = torch.cat(batch)
        # return flattened_targets, target_lengths
        return pad_sequence(batch, batch_first=True, padding_value=self.blank_id), target_lengths

    def _filter(self, probs: Tensor, ids: Tensor) -> tuple[Tensor, List[int]]:
        ids = list(zip(*groupby(ids.tolist())))[0]  # remove duplicate
        ids = [x for x in ids if x != self.blank_id]
        return probs, ids
    # def decode(self, token_dists: Tensor, beam_width: int = 10) -> tuple[List[List[str]], List[float]]:
    #     """
    #     Decode một batch xác suất token thành gloss bằng beam search.
    #     Args:
    #         token_dists: softmax probs, shape N, L, C
    #         decoder: pyctcdecode decoder đã được khởi tạo với vocab
    #         beam_width: số beam giữ lại
    #     Returns:
    #         List gloss (danh sách từ) và list log-probs
    #     """
    #     batch_tokens = []
    #     batch_log_probs = []
    #     for dist in token_dists:
    #         # Chuyển sang numpy array [T, C]
    #         dist_np = dist.cpu().detach().numpy()
    #         # Decode với beam search
    #         # text = self.decoder.decode(dist_np)
    #         # print(text)
    #         # input()
    #         decoded = self.decoder.decode_beams(dist_np, beam_width=beam_width)
    #         # Chọn chuỗi tốt nhất
    #         best = decoded[0]
    #         text = best.text.split()  # List[str], chia theo từ
    #         log_prob = best.logit_score  # log-score tổng

    #         batch_tokens.append(text)
    #         batch_log_probs.append(log_prob)
        
    #     return batch_tokens, batch_log_probs





def build_tokenizer_from_csv(label_csv_files, tokenizer_type='ctc', extra_vocab=None):
    """
    Xây tokenizer gloss từ toàn bộ label trong nhiều file csv.
    Args:
        label_csv_files: list các path tới file csv chứa id và gloss/annotation
        tokenizer_type: 'ctc' hoặc 'seq2seq'
        extra_vocab: list từ bổ sung (nếu có)
    Returns:
        tokenizer gloss phù hợp
    """

    all_glosses = set()
    for file in label_csv_files:
        all_data = pd.read_csv(file, delimiter=",")
        all_data = all_data[all_data["id"].notna()]
        all_data = all_data[all_data["gloss"].notna()]
        label_col = "gloss"
        for annotation in all_data[label_col]:
            tokens = str(annotation).split()
            all_glosses.update(tokens)
    if extra_vocab is not None:
        all_glosses.update(extra_vocab)
    vocab_list = sorted(all_glosses)
    if tokenizer_type == 'ctc':
        tokenizer = CTCGlossTokenizer(vocab_list)
    else:
        tokenizer = GlossTokenizer(vocab_list)
    return tokenizer 