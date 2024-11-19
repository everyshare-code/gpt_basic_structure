import torch
import torch.nn as nn
from transformer_block import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 1024, d_model))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)

    def forward(self, x, mask=None):
        # 토큰 임베딩과 위치 임베딩 결합
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding[:, :x.size(1), :]
        x = token_embeddings + position_embeddings

        # Transformer 블록 통과
        for block in self.transformer_blocks:
            x = block(x, mask)

        x = self.norm(x)
        return self.output(x)