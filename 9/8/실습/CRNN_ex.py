from cv2 import transform
import torch
import torch.nn as nn 
import torch.optim as optim 
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader 
import torchvision.transforms as transforms 
import numpy as np 
import matplotlib.pyplot as plt
from PIL import Image, ImageDraw, ImageFont 
import string 
import random


class CRNN(nn.Module):
    def __init__(self, img_height, img_width, num_chars, num_classes, rnn_hidden=256) -> None:
        super().__init__()

        self.img_height = img_height
        self.img_width = img_width
        self.num_chars = num_chars

        # CTC용 클래스 수
        # blank: 빈 문자 토큰
        # EOS: 문자열 종료 토큰
        self.num_classes = num_classes

        # CNN 백본
        # 백본 네트워크: 이미지에서 시각적 특징을 추출하는 합성곱 신경망
        self.cnn = nn.Sequential(
            # layer 1 - 기본 특징 추출
            # 입력 채널 수, 출력 채널 수, 필터 크기, 필터 이동 거리, 패딩 크기
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            # 32 x 128 -> 16 x 64
            # 해상도 절반으로 축소
            nn.MaxPool2d(2, 2),

            # layer 2 - 더 복잡한 특징 추출
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            # 16 x 64 -> 8 x 32
            nn.MaxPool2d(2, 2),

            # layer 3 - 고수준 특징 추출
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),

            # layer 4 - 특징 정제
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            # 8 x 32 -> 4 x 32
            nn.MaxPool2d((2, 1)),

            # layer 5 - 더 깊은 특징 추출
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),

            # layer 6 - 특징 강화
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            # 4 x 32 -> 2 x 32
            nn.MaxPool2d((2, 1)),

            # layer 7 - 최종 측징 앱 생성
            # 2 x 32 -> 1 x 31
            nn.Conv2d(512, 512, 2, 1, 0), nn.ReLU()
        )

        # RNN
        # 텍스트 순서 정보를 처리하는 LSTM
        # 512: 마지막 CNN 출력 채널 수, 
        # rnn_hidden: 은닉 상태 차원 수,
        #   - 실제 LSTM 내부에는 4개의 게이트가 있다.(forget, input, output, cell)
        #   - 각 게이트마다 rnn_hidden 크기의 가중치 행렬이 필요
        #   - 총 파라미터 수 = 4 x (input_size + hidden_size) x hiddne_size
        #   - 예) 4 x (256 + 256) x 256 = 786,432 파라미터(단방향 기중)
        # bidirectional: 양방향 LSTM(forward, backward), 
        #   - 파라미터 수가 2배가 된다.(각 방향마다 별도 LSTM)
        #   - 출력 지원: 2 x rnn_hidden = 512
        # batch_first: 입력 형태[batch, sequence, features]
        #   - 데이터 배치를 첫 번째 지원으로 사용
        #   - 예) [batch, sequence, features] -> [batch, sequence, features]
        #   - 일반적인 RNN 입력 생성: [sequence, batch, features]를 CNN 출력과 만추기 위해 변환
        self.rnn = nn.LSTM(512, rnn_hidden, bidirectional=True, batch_first=True)

        # 출력 레이어 - RNN 출력을 문자 클래스로 변환
        self.linear = nn.Linear(rnn_hidden * 2, num_classes)

    def forward(self, x):
        # [batch, 512, 1, width]
        conv_features = self.cnn(x)

        # batch_size, channels, height, width
        b, c, h, w = conv_features.size()

        # 텐서의 값은 유지, 차원 변경
        # CNN 출력의 height는 보통 1
        # width는 시퀀스 길이로 사용
        # height x channels = 각 시퀀스 위치에서의 feature 벡터 지원
        conv_features = conv_features.view(b, c*h, w)
        # [batch, 512, width]

        # 차원 순서 변경
        # 0번째 차원은 유지, 1번째는 2번째로, 2번째는 1번째로 이동
        conv_features = conv_features.permute(0, 2, 1)
        # 예) [batch, 512, 25] -> [batch, 25, 512]

        # 순차 정보 처리
        # rnn_out: (batch_size, width, hidden * 2)
        # bidirectional=True 이므로 hidden * 2
        rnn_out, _ = self.rnn(conv_features)

        # 출력 레이어
        # [batch, width, num_classes]
        # 각 문자 위치에 대해 num_classes 중 하나 예측
        output = self.linear(rnn_out)

        # dim=2: 마지막 차원(클래스 차원)에 대해 소프트맧 적용
        # output: (batch_size, width, num_classes)
        output = F.log_softmax(output, dim=2)

        return output 


class SyntheticTextDataset(Dataset):
    def __init__(self, num_samples=1000, img_height=32, img_width=128, max_text_len=6) -> None:
        self.num_samples = num_samples
        self.img_height = img_height
        self.img_width = img_width
        self.max_text_len = max_text_len

        # 문자 집합 정의 0~9
        self.chars = string.digits

        # 문자를 인덱스로 매핑
        self.char_to_idx = {char: idx+1 for idx, char in enumerate(self.chars)}
        
        # CTC blank 토큰
        self.char_to_idx['<blank>'] = 0

        # 인덱스를 문자로 역매핑
        self.idx_to_char = {idx:char for char, idx in self.char_to_idx.items()}

        # CTC용 클래스 개수(문자 수 + blank)
        self.num_classes = len(self.chars) + 1

        # num_classes: 문자 개수 + blank + EOS
        # 예) 26 + 1 +1 = 28(소문자 26 + 빈 문자 1 + 문자열 종료 1)

        # 이미지 전처리 변환
        self.transform = transforms.Compose([
            # PIL Image -> Tensor
            transforms.ToTensor(),
            # [-1, 1] 범위로 정규화
            transforms.Normalize((0.5,), (0.5,))
        ])

    def __len__(self):
        return self.num_samples
    
    def __getitem__(self, idx):
        # 랜덤 텍스트 생성
        text_len = random.randint(2, min(4, self.max_text_len))

        # random.choices(self.chars, k=text_len): 문자 집합에서 text_len 길이만큼 문자 선택
        # ''.join(random.choices(self.chars, k=text_len)): 선택된 문자들을 문자열로 결합
        text = ''.join(random.choices(self.chars, k=text_len))

        image = self.create_text_image(text)

        # 텍스트를 인덱스 라벨로 변환
        label = [self.char_to_idx[char] for char in text]

        if self.transform:
            image = self.transform(image)

        return image, torch.tensor(label, dtype=torch.long), text

    def create_text_image(self, text):
        # 그레이스케일 이미지, 이미지 크기, 흰색
        img = Image.new('L', (self.img_width, self.img_height), 255)

        # 이미지에 텍스트를 그리기 위한 객체
        draw = ImageDraw.Draw(img)

        try:
            font = ImageFont.truetype('arial.ttf', 24)
        except:
            font = ImageFont.load_default()

        try:
            bbox = draw.textbbox((0, 0), text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
        except AttributeError:
            text_width, text_height = draw.textsize(text, font=font)

        # 중앙 정렬로 텍스트 그리기
        x = (self.img_width - text_width) // 2
        y = (self.img_height - text_height) // 2

        draw.text((x, y), text, fill=0, font=font)

        # 노이즈 추가
        img_array = np.array(img)
        # 가우시안 노이즈
        noise = np.random.normal(0, 5, img_array.shape)

        # np.clip(img_array + noise, 0, 255): 노이즈를 추가한 이미지를 0과 255 사이로 클리핑(제한)
        # .astype(np.uint8): 이미지를 8비트 이미지로 변환
        img_array = np.clip(img_array + noise, 0, 255).astype(np.uint8)

        return Image.fromarray(img_array)


class CRNNTrainer:
    def __init__(self, model, device='cpu'):
        self.model = model.to(device)
        self.device = device

        # CTC Loss 함수: 문자인식에서 주로 사용하는 손실함수
        # blank: 토큰 인덱스 0, zero_infinity: 무한대 loss 값을 0으로 처리
        self.criterion = nn.CTCLoss(blank=0, reduction='mean', zero_infinity=True)

        # Adam: 학습률 감소 및 정규화 추가
        # lr 학습률, weight_decay: L2 정규화, model.parameters(): 모델의 파라미터 가져오기
        self.optimizer = optim.Adam(model.parameters(), lr=0.0005, weight_decay=1e-4)

        # StepLR 스케츌러의 안정성 향상 원리:
        # setp_size=3 : 3 에포크마다 학습률 감소
        # gamma=0.5 : 학습률을 0.5배로 감소
        # 학습 초기: 큰 학습률로 빠른 학습
        # 학습 후기: 작은 학습률로 세밀한 조정
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=3, gamma=0.5)
        
    def train_epoch(self, dataLoader):
        self.model.train()

        total_loss = 0
        num_batches = 0

        for batch_idx, (images, targets, texts) in enumerate(dataLoader):
            # 데이터를 디바이스로 이동
            images = images.to(self.device)

            # 라벨 데이터 리스트, 각 텐서를 개별적으로 디바이스로 이동
            targets = [target.to(self.device) for target in targets]

            # forward pass
            # 그래디언트 초기화
            self.optimizer.zero_grad()

            outputs = self.model(images)

            outputs = outputs.permute(1, 0, 2)

            # image_lengths : 모델이 출력한 시퀀스 길이
            # torch.full( ~ ... dtype=torch.long) : 모든 샘플의 입력 길이를 채우는 텐서 생성
            # images.size(0) : 이미지 데이터의 베치 크기
            # ouputs.size(0) : 모든 출력의 시퀀스 길이
            # dtype=torch.long : 데이터 타입을 정수형으로 변환
            input_lengths = torch.full((images.size(0),), outputs.size(0), dtype=torch.long)

            # 실제 정답 텍스트의 길이
            target_lengths = torch.tensor([len(target) for target in targets], dtype=torch.long)

            targets_id = torch.cat(targets)

            # CTC loss 계산
            # self.criterion : CTC loss 함수
            #  outputs : 모델 출력
            #  targets_id : 타겟을 1차원으로 연결
            # input_lengths : 모든 샘플의 입력 길이
            #  target_lengths : 각 타겟의 길이
            loss = self.criterion(outputs, targets_id, input_lengths, target_lengths)

            # backward
            loss.backward()

            # 그래디언트 클리핑 추가
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)

            # 가중치 업데이트
            self.optimizer.step()

            total_loss += loss.item()
            num_batches += 1
            
            if batch_idx % 50 == 0:
                print(f"Batch {batch_idx}, Loss: {loss.item():.4f}")
        
        self.scheduler.step()
        return total_loss / num_batches

    def decode_prediction(self, output, dataset):
        # torch.argmax(output, dim=2) : 각 시점에서 가장 확률이 높은 문자 선택
        # ouput : 모델 출력
        # dim=2 : 마지막 차원에 대해 최대값 인덱스 찾기
        # pred_indices : 각 시점에서 가장 확률이 높은 문자 선택
        pred_indices = torch.argmax(output, dim=2)

        decoded_texts = []
        blank_ratios = []

        for batch_idx in range(pred_indices.size(0)):
            # 텐서를 넘파이 배열로 변환
            # cpu() : 텐서를 cpu로 이동
            indices = pred_indices[batch_idx].cpu().numpy()

            decoded_chars = []
            prev_idx = -1
            blank_count = 0

            for idx in indices:
                # blank 가 아니고 이전고 다른 경우
                if idx != 0 and idx != prev_idx:
                    if idx in dataset.idx_to_char:
                        decoded_chars.append(dataset.idx_to_char[idx])
            
                prev_idx = idx
                if idx == 0:
                    blank_count += 1

            decoded_texts.append(''.join(decoded_chars))
            blank_ratios.append(blank_count / len(indices) if len(indices) > 0 else 1.0)

        return decoded_texts, blank_ratios

    def evaluate(self, dataLoader, dataset, num_samples=10):
        # 평가 모드 설정
        self.model.eval()
        correct = 0
        total = 0

        # 그래디언트 계산 비활성화
        with torch.no_grad():
            for batch_idx, (images, targets, gt_texts) in enumerate(dataLoader):
                if batch_idx >= num_samples:
                    break
                images = images.to(self.device)
                outputs = self.model(images)

                predicted_texts, blank_ratios = self.decode_prediction(outputs, dataset)

                for pred, gt, br in zip(predicted_texts, gt_texts, blank_ratios):
                    if pred == gt:
                        correct +=1
                    total += 1

                    print(f"GT: '{gt}' | Pred: '{pred}' | BlankRatio: {br:.2f} | {'O' if pred == gt else 'X'}")
            
        accuracy = correct / total if total > 0 else 0
        print(f"\nAccuracy: {accuracy:.4f} ({correct}/{total})")
        return accuracy

def crnn_practice():
    print("=== CRNN OCR 실습 ===")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print(f"사용 다바이스: {device}")

    print(f"\n1. 합성 데이터셋 생성")
    # 데이터셋 생성
    train_dataset = SyntheticTextDataset(num_samples=1000, img_height=32, img_width=128)
    test_dataset = SyntheticTextDataset(num_samples=100, img_height=32, img_width=128)

    # 데이터 로더 생성
    # DataLoader : 데이터셋을 배치 단위로 관리하고 반복 가능한 객체로 변환
    # batch_size=32 : 배치크기
    # shuffle : 데이터 섞기
    # collate_fn=collate_fn : 배치 데이터를 적절한 형태로 조합
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=True, collate_fn=collate_fn)

    print(f'훈련 데이터: {len(train_dataset)}개, 테스트 데이터: {len(test_dataset)}개')
    print(f"문자 집합: {train_dataset.chars}")

    # 시각화
    sample_image, sample_label, sample_text = train_dataset[0]
    plt.rc('font', family='Apple SD Gothic Neo')
    plt.figure(figsize=(10,3))
    plt.imshow(sample_image.squeeze(), cmap='gray')
    plt.title(f"샘플 이미지: '{sample_text}'")
    plt.axis('off')
    plt.show()

    # 모델 생성
    print("\n2. CRNN 모델 생성")
    model = CRNN(
        img_height=32,
        img_width=128,
        num_chars=len(train_dataset.chars),
        num_classes=train_dataset.num_classes
    )

    print(f"모델 파라미터 수: {sum(p.numel() for p in model.parameters()):,}")

    # 훈련
    print("\n3. 모델 훈련")
    trainer = CRNNTrainer(model, device)

    num_epochs = 5
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        avg_loss = trainer.train_epoch(train_loader)

        # 옵티마이저의 학습률
        # 옵티마이저의 파라미터 그룹
        # lr: 학습률
        current_lr = trainer.optimizer.param_groups[0]['lr']

        print(f"평균 손실: {avg_loss:.4f}, 학습률: {current_lr:.6f}")

        if (epoch + 1) % 2 == 0:
            print(f"\n{epoch+1} 에포크 평가:")
            trainer.evaluate(test_loader, test_dataset, num_samples=5)

    # 최종 평가
    print(f"\n4. 최종 평가")
    final_accuracy = trainer.evaluate(test_loader, test_dataset, num_samples=20)
    return model, train_dataset, test_dataset

def collate_fn(batch):
    images, labels, texts = zip(*batch)

    # 이미지는 동일한 크기이므로 스택 
    images = torch.stack(images)

    # 길이가 다르므로 리스트
    labels = list(labels)
    return images, labels, texts


model, train_dataset, test_dataset = crnn_practice()