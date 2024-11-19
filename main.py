import torch
from gpt_model import GPTModel

# 모델 사용 예시
def main():
    # 모델 초기화
    model = GPTModel(
        vocab_size=50000,
        d_model=768,
        num_layers=12,
        num_heads=12,
        d_ff=3072
    )

    # 테스트용 입력 생성
    batch_size = 2
    seq_length = 16
    input_ids = torch.randint(0, 50000, (batch_size, seq_length))

    print("\n=== GPT 모델 테스트 ===")

    # 1. 모델 통계 출력
    stats = model.get_model_stats()
    print("\n모델 구조:")
    for key, value in stats.items():
        print(f"{key}: {value}")

    # 2. 임베딩 테스트
    emb_test = model.embedding(input_ids)
    print("\n임베딩 테스트:")
    for key, value in emb_test.items():
        print(f"{key} shape: {value.shape}")

    # 3. 어텐션 테스트
    att_test = model.attention(input_ids)
    print("\n어텐션 테스트:")
    print(f"Attention output shape: {att_test['attention_shape']}")

    # 4. 전체 forward 패스 테스트
    forward_test = model.full_forward(input_ids)
    print("\nForward 패스 테스트:")
    print(f"Final output shape: {forward_test['final_output'].shape}")


if __name__ == "__main__":
    main()