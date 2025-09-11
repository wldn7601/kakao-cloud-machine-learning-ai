import torch
from transformers import BertTokenizer, BertForSequenceClassification
import pandas as pd
from sklearn.preprocessing import LabelEncoder

class IntentClassifier:
    def __init__(self, model_name='klue/bert-base'):
        # 의도 분류기 초기화
        # 텍스트를 BERT가 이해할 수 있는 토큰으로 변환
        self.tokenizer = BertTokenizer.from_pretrained(model_name)

        # 훈션 메서드에사 초기화 됨
        self.model = None

        # 텍스트 라벨을 숫자로 변환한다.
        # BERT 모델이 처리할 수 있게 한다.
        self.label_encoder = LabelEncoder()

    # 텍스트를 토큰화, 라벨을 숫자로 변환
    def prepare_data(self, texts, labels):
        # 텍스트 라벨을 숫자로 변환, numpy.ndarray 형태
        # 텍스트의 의도를 인덱스로 바꿈
        # 예) '인사' -> 0, '날씨' -> 1 ...
        encoded_labels = self.label_encoder.fit_transform(labels)

        # 텍스트를 토큰화한다.
        # 딕셔너리 형태로 여러 텐서를 포함
        encodings = self.tokenizer(
            texts,
            # 최대 길이 초과시 잘라냄
            truncation=True,
            # 배치 내 모든 시퀀스를 같은 길이로 맞춤
            padding=True,
            # 최대 토큰 길이
            max_length=128,
            # PyTorch 텐셔로 변환
            return_tensors='pt'
        )
        print(f"encodings: {encodings}")
        print(f"encoded_labels: {encoded_labels}")
        return encodings, encoded_labels

    def train(self, train_texts, train_labels):
        # 라벨의 중복을 제거한다.
        num_labels = len(set(train_labels))

        # 모델 초기화
        self.model = BertForSequenceClassification.from_pretrained(
            'klue/bert-base',
            # 분류할 의도 개수만큼 출력 뉴런 설정
            num_labels=num_labels
        )

        # 훈련 데이터 준비
        train_encodings, train_labels_encoded = self.prepare_data(train_texts, train_labels)

        # 커스텀 데이터셋 클래스 정의
        # pytorch 데이터셋 생성
        # 데이터 셋 클래스를 만드느 이유:
        # - PyTorch의 DataLoader 시스템 때문이다.
        # - 이렇게 직접 사용하면 안 됨
        # - train_loader = DataLoader(train_encodings, batch_size=16)  # ❌ 에러!
        # - PyTorch DataLoader는 Dataset 객체만 받음
        # - train_loader = DataLoader(train_dataset, batch_size=16)    # ✅ 정상
        class IntentDataset(torch.utils.data.Dataset):
            def __init__(self, encodings, labels):
                # 데이터 초기화
                self.encodings = encodings
                self.labels = labels
            
            def __getitem__(self, idx):
                # 토큰화된 정보를 가져와 해당 인덱스의 정보를 
                # 텐서로 변환하여 가져온다.
                item = {key:torch.tensor(val[idx]) for key, val in self.encodings.items()}
                # 숫자로 변환된 라벨을 가져와 해당 인덱스의 정보를
                # 텐서로 변환하여 가져온다
                item['labels'] = torch.tensor(self.labels[idx])

                return item
            
            def __len__(self):
                # 
                return len(self.labels)

        train_dataset = IntentDataset(train_encodings, train_labels_encoded)

        # 훈련 설정(기존 버전 호환성을 위해 직접 구현)

        # 한번에 처리할 데이터 샘플 수
        # 큰 배치: 안정적인 그래디언트, 빠른 학습, 메모리 많이 사용
        # 작은 배치: 노이즈가 잇는 그래디언트, 더 나은 일반화, 메모리 절약
        batch_size = 16

        # 전체 훈련 데이터를 3번 반복
        # 한번의 훈련으로는 충분히 학습할 수 없다.
        # 너무 많이 반복하면 과적합 문제 발생
        epochs = 3

        # 학습률
        # 큰 학습률: 불안정한 학습
        # 작은 학습률: 느린 학습
        learning_rate = 2e-5

        # L2 정규화 강도
        # 예) L2 정규화 사용 x
        # Loss = CrossEntropyLoss(predictions, labels)
        # 예) L2 정규화 사용
        # Loss = CrossEntropyLoss(predictions, labels) + weight_dency * Σ(weight²)
                                                    #       ↑
                                                    # L2 정규화 항
        # 따라서 과적합을 방지해 준다.
        weight_decay = 0.01

        # 데이터 로더 생성
        # 데이터 로더는 학습을 위한 데이터 공급 파이프 라인 역할을 한다.
        train_dataloader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            # 과적합을 방지하기 위해 데이터 섞기
            shuffle=True
        )

        # 옵티마이저 설정
        # 옵티마이저는 모델의 가중치를 업데이트하는 알고리즘이다.
        # AdamW = Adam + Weight Decay, BERT 훈련에 최적화된 옵티마이저
        # Adam = Adaptive Moment Estimation
        # - 기울기가 큰 파라미터 -> 작은 학습률
        # - 기울기가 작은  -> 큰 학습률

        # 적응형 학습률 예)
        # param1.grad = 10.0    # 큰 기울기
        # param2.grad = 0.1     # 작은 기울기

        # # Adam이 자동으로 조정:ㅇ
        # param1_lr = 0.001 * (1/large_factor)  # 실제로는 작게 업데이트
        # param2_lr = 0.001 * (1/small_factor)  # 실제로는 크게 업데이트

        # 모멘텀 효과
        # 이전 기울기들의 누적 효과
        # momentum = 0.9 * prev_momentum + 0.1 * current_gradient
        # # 효과: 일관된 방향으로 더 빠르게 수렴

        optimizer = torch.optim.AdamW(
            # 해당 모델의 모든 학습 가능한 가중치, 편향을 반환한다.
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay
        )

        # PyTorch 모델은 두가지 모드를 가진다.
        # 훈련모드: model.train()
        # 평가모드: model.eval()
        self.model.train()

        
        for epoch in range(epochs):
            print(f"Epoch {epoch+1}/{epochs}")
            # 각 반복마다 평균 로스 계산
            total_loss = 0

            # 해당 반복문에서 batch 형태
            # batch = {
            #     'input_ids': tensor([
            #         [101, 12543, 8921, 102, 0, 0, ...],      # 문장 1: "안녕하세요"
            #         [101, 9876, 5432, 3456, 789, 102, ...],  # 문장 2: "오늘 날씨 어때요?"
            #         [101, 11111, 2222, 102, 0, 0, ...]       # 문장 3: "감사합니다"
            #         # ... 총 16개 문장
            #     ]),  # shape: [16, 128]
                
            #     'attention_mask': tensor([
            #         [1, 1, 1, 1, 0, 0, ...],  # 실제 토큰=1, 패딩=0
            #         [1, 1, 1, 1, 1, 1, ...],
            #         [1, 1, 1, 1, 0, 0, ...]
            #         # ... 총 16개
            #     ]),  # shape: [16, 128]
                
            #     'token_type_ids': tensor([
            #         [0, 0, 0, 0, 0, 0, ...],  # 보통 모두 0 (단일 문장)
            #         [0, 0, 0, 0, 0, 0, ...],
            #         [0, 0, 0, 0, 0, 0, ...]
            #     ]),  # shape: [16, 128]
                
            #     'labels': tensor([1, 2, 0, 1, 2, 0, ...])  # 각 문장의 의도 레이블
            # }  # shape: [16]
            for batch in train_dataloader:
                # 그래디언트 초기화
                # PyTorch는 기본적으로 기울기를 누적한다.
                # 따라서 반복 훈련시 초기마다 초기화해준다.
                optimizer.zero_grad()
                
                # 순전파
                # **는 언패킹 연산자다.
                # **batch는 batch의 키워드 인자들이다.

                # 위의 예를 들면
                # outputs = self.model(
                #     input_ids=batch['input_ids'],
                #     attention_mask=batch['attention_mask'], 
                #     labels=batch['labels']
                # )
                outputs = self.model(**batch)
                loss = outputs.loss

                # 역전파
                loss.backward()

                # 기울기 클리핑
                # 기울기가 너무 클 때 설정된 최대값으로 제한한다. -> 기울기 폭파 방지
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                # 가중치 업데이트
                optimizer.step()

                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_dataloader)
            print(f"Average Loss: {avg_loss:.4f}")
        
        print("훈련 완료")

    # 의도 예측
    def predict(self, text):
        # 텍스트를 BERT가 이해할 수 있는 토큰으로 변환
        # PyTorch 형태의 텐서로 반환
        inputs = self.tokenizer(text, return_tensors='pt', truncation=True, padding=True)

        print(f"inputs: {inputs}")

        # 예측시에는 그래디언트 계산 불필요
        with torch.no_grad():
            # inputs 딕셔너리의 키워드들로 언팩킹
            ouputs = self.model(**inputs)

            # BERT 모델의 원시 출력 (로짓)
            # outputs.logits = tensor([
            #     [2.1, -0.5, 1.3],    # 첫 번째 문장의 각 클래스별 점수
            #     [0.8, 3.2, -1.1],    # 두 번째 문장의 각 클래스별 점수  
            #     [-0.3, 1.5, 2.7]     # 세 번째 문장의 각 클래스별 점수
            # ])  # shape: [batch_size, num_classes] = [3, 3]

            # # 점수 의미 (예시):
            # # [:, 0] → '인사' 클래스 점수
            # # [:, 1] → '질문' 클래스 점수  
            # # [:, 2] → '감사' 클래스 점수

            # dim=1 : 행 방향을 기준으로 softmax 적용
            predictions = torch.nn.functional.softmax(ouputs.logits, dim=1)

        # 행을 기준으로 가장 확률이 높은 인덱스를 가져온다.
        predicted_class = torch.argmax(predictions, dim=1).item()
        
        # 예측 신뢰도는 확률이 가장 높은 값
        confidence = torch.max(predictions).item()

        # 숫자 라벨을 원래 텍스트 라벨로 변환
        # 가장 확률이 높은 인덱스를 가져와서 텍스트 라벨로 변환
        intent = self.label_encoder.inverse_transform([predicted_class])[0]

        return intent, confidence

# 사용 예시 (실제 데이터 필요)

# 샘플 데이터
train_texts = [
    "안녕하세요", "반갑습니다", "hello",
    "날씨가 어때요?", "비가 와요?", "맑나요?",
    "주문하고 싶어요", "메뉴 보여줘", "배달 가능해?"
]

train_labels = [
    "greeting", "greeting", "greeting",
    "weather", "weather", "weather", 
    "order", "order", "order"
]
# 각 텍스트에 해당하는 의도 라벨
# greeting: 인사, weather: 날씨 문의, order: 주문 관련

classifier = IntentClassifier()
# 의도 분류기 인스턴스 생성

classifier.train(train_texts, train_labels)
# 샘플 데이터로 모델 훈련

# 예측
intent, confidence = classifier.predict("오늘 날씨 어때요?")
# 새로운 텍스트의 의도 예측

print(f"의도: {intent}, 신뢰도: {confidence:.2f}")
# 예측 결과 출력 (의도와 신뢰도)