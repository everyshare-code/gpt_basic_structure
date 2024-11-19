import torch
import torch.nn as nn
from transformer_block import TransformerBlock

class GPTModel(nn.Module):
    def __init__(self, vocab_size, d_model=768, num_layers=12, num_heads=12, d_ff=3072, dropout=0.1):
        super().__init__()
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        self.position_embedding = nn.Parameter(torch.zeros(1, 1024, d_model))

        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(d_model, num_heads, d_ff, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(d_model)
        self.output = nn.Linear(d_model, vocab_size)
    def create_attention_mask(self, size):
        """GPT의 캐주얼 마스크 생성 (미래 토큰을 보지 못하게 함)"""
        mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
        return ~mask
    def forward(self, x, mask=None):
        token_embeddings = self.token_embedding(x)
        position_embeddings = self.position_embedding[:, :x.size(1), :]
        x = token_embeddings + position_embeddings

        for block in self.transformer_blocks:
            x = block(x, mask)

        x = self.norm(x)
        return self.output(x)

    def embedding(self, input_ids):
        """임베딩 레이어 테스트"""
        token_emb = self.token_embedding(input_ids)
        pos_emb = self.position_embedding[:, :input_ids.size(1), :]
        combined_emb = token_emb + pos_emb
        return {
            'token_embedding': token_emb,
            'position_embedding': pos_emb,
            'combined_embedding': combined_emb
        }

    def attention(self, input_ids):
        """어텐션 메커니즘 테스트"""
        embeddings = self.embedding(input_ids)['combined_embedding']
        attention_outputs = []

        # 첫 번째 트랜스포머 블록의 어텐션 패턴 확인
        block = self.transformer_blocks[0]
        mask = self.create_attention_mask(input_ids.size(1))
        attention_output = block.attention(embeddings, embeddings, embeddings, mask)

        return {
            'attention_pattern': attention_output,
            'attention_shape': attention_output.shape
        }
    def full_forward(self, input_ids):
        """전체 forward 패스 테스트"""
        mask = self.create_attention_mask(input_ids.size(1))

        # 각 단계별 출력 저장
        outputs = {}

        # 1. 임베딩
        embeddings = self.embedding(input_ids)
        outputs['embeddings'] = embeddings

        # 2. 트랜스포머 블록
        x = embeddings['combined_embedding']
        block_outputs = []
        for i, block in enumerate(self.transformer_blocks):
            x = block(x, mask)
            block_outputs.append(x)
        outputs['transformer_blocks'] = block_outputs

        # 3. 최종 출력
        final_output = self.output(self.norm(x))
        outputs['final_output'] = final_output

        return outputs

    def get_model_stats(self):
        """모델 구조 및 파라미터 통계"""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            'vocab_size': self.vocab_size,
            'd_model': self.d_model,
            'num_layers': self.num_layers,
            'num_heads': self.num_heads,
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024)
        }