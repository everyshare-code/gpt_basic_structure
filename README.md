# GPT 기본 구조 구현

# 프로젝트 설명:
이 프로젝트는 PyTorch를 사용하여 구현된 GPT(Generative Pre-trained Transformer)의 기본 구조입니다. 모델의 동작을 테스트하고 검증하기 위한 다양한 테스트 기능을 포함하고 있습니다.

# 주요 기능:
모델은 다음과 같은 테스트 기능들을 제공합니다:

1. 모델 구조 및 통계 확인 (get_model_stats)
   - 총 파라미터 수
   - 학습 가능한 파라미터 수
   - 모델 크기
   - 주요 하이퍼파라미터

2. 임베딩 레이어 테스트 (embedding)
   - 토큰 임베딩
   - 위치 임베딩
   - 결합된 임베딩

3. 어텐션 메커니즘 테스트 (attention)
   - 어텐션 패턴 분석
   - 출력 형태 확인

4. 전체 Forward 패스 테스트 (full_forward)
   - 최종 출력 형태 검증

# 사용 예시:
```python
import torch
from gpt_model import GPTModel

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

# 모델 테스트 실행
stats = model.get_model_stats()
emb_test = model.embedding(input_ids)
att_test = model.attention(input_ids)
forward_test = model.full_forward(input_ids)
```

# 설치 및 의존성:
```
torch
numpy
```

# 프로젝트 구조:
```
gpt_basic_structure/
│
├── main.py                  # 모델 테스트 실행 스크립트
├── gpt_model.py            # GPT 모델 클래스 정의
├── transformer_block.py    # 트랜스포머 블록 구현
├── feed_forward.py        # Feed Forward 네트워크
├── multi_head_attention.py # 멀티헤드 어텐션 구현
└── requirements.txt        # 의존성 패키지 목록
```

# 개발자 정보:
- GitHub: [everyshare-code](https://github.com/everyshare-code)
- 이메일: park20542040@gmail.com