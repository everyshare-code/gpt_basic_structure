import torch
from gpt_model import GPTModel

def create_attention_mask(size):
    """GPT의 캐주얼 마스크 생성 (미래 토큰을 보지 못하게 함)"""
    mask = torch.triu(torch.ones(size, size), diagonal=1).bool()
    return ~mask


# 모델 사용 예시
def main():
    # 모델 파라미터
    vocab_size = 50000
    d_model = 768
    num_layers = 12
    num_heads = 12
    d_ff = 3072

    # 모델 초기화
    model = GPTModel(
        vocab_size=vocab_size,
        d_model=d_model,
        num_layers=num_layers,
        num_heads=num_heads,
        d_ff=d_ff
    )

    # 입력 데이터 예시
    batch_size = 4
    seq_length = 32
    x = torch.randint(0, vocab_size, (batch_size, seq_length))
    mask = create_attention_mask(seq_length)

    # 모델 실행
    output = model(x, mask)
    print(f"입력 형태: {x.shape}")
    print(f"출력 형태: {output.shape}")
    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters())}")


if __name__ == "__main__":
    main()