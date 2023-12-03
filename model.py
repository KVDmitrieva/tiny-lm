import math

import torch
import numpy as np
from torch import nn
from torch import Tensor
from torch.distributions.categorical import Categorical


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len: int = 5000):
        super().__init__()

        pos = torch.arange(0, max_len).unsqueeze(1)
        emb = torch.arange(0, embed_dim, 2)
        emb = np.exp(-emb * np.log(10000) / embed_dim)

        pe = torch.zeros(max_len, embed_dim)

        pe[:, 0::2] = torch.sin(pos * emb)
        pe[:, 1::2] = torch.cos(pos * emb)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        batch_size, cur_len, _ = x.shape
        return x + self.pe.repeat(batch_size, 1, 1)[:, :cur_len]


class MightyLanguageModel(nn.Module):
    def __init__(self, vocab_size, max_len, pad_idx, n_layers=1,
                 embed_dim=128, n_head=4, hidden_dim=256, dropout=0.1):
        super().__init__()
        self.pad_idx = pad_idx
        self.max_len = max_len
        self.embed_dim = embed_dim

        enc_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim, nhead=n_head, dim_feedforward=hidden_dim,
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.pos_enc = PositionalEncoding(embed_dim, max_len)
        self.encoder = nn.TransformerEncoder(enc_layer, num_layers=n_layers)
        self.classifier = nn.Linear(embed_dim, vocab_size)

    def forward(self, tokens: Tensor) -> Tensor:
        x = self.embedding(tokens) * math.sqrt(self.embed_dim)
        x = self.pos_enc(x)
        x = self.encoder(x, src_key_padding_mask=tokens == self.pad_idx)
        return self.classifier(x)

    @torch.inference_mode()
    def inference(self, dataset, prefix: str = '', temp: float = 1.) -> str:
        self.eval()
        device = next(self.parameters()).device

        enc_prefix = [dataset.bos_id] + dataset.sp_model.encode(prefix)
        tokens = torch.tensor(enc_prefix).unsqueeze(0).to(device)

        while tokens.shape[1] < self.max_len:
            emb = self.pos_enc(self.embedding(tokens) * math.sqrt(self.embed_dim))
            logits = self.classifier(self.encoder(emb)) / temp
            new_tokens = Categorical(logits=logits[:, -1:]).sample()
            tokens = torch.cat([tokens, new_tokens], dim=1)
            if new_tokens.item() == dataset.eos_id:
                break

        generated = dataset.ids2text(tokens.squeeze())
        return generated

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)


