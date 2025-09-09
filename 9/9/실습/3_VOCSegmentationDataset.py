"""
=== VOC 세그멘테이션 데이터셋 처리 및 시각화 예제 ===

💡 핵심: "세그멘테이션이 뭔지 보여주는" 데모 코드!
- 세그멘테이션 결과가 어떻게 생겼는지 시각적으로 체험
- 원본 이미지 vs 픽셀별 클래스 분할 결과를 나란히 비교
- "아, 세그멘테이션이 이런 거구나!" 하고 이해할 수 있는 교육용 코드
- FCN, U-Net 등 실제 모델 학습은 별도 파일에서 진행

이 예제의 목표:
1. 의미론적 세그멘테이션(Semantic Segmentation)이 무엇인지 이해
2. PASCAL VOC 2012 데이터셋의 구조와 특성 파악
3. 이미지와 세그멘테이션 마스크의 전처리 방법 차이점 학습
4. 실제 세그멘테이션 데이터가 어떻게 생겼는지 시각적으로 확인
5. 딥러닝 세그멘테이션 모델 학습을 위한 데이터 준비 과정 이해

보여주는 내용:
- 원본 이미지: 일반적인 RGB 사진 (사람, 동물, 차량 등이 포함된 자연스러운 장면)
- 세그멘테이션 마스크: 픽셀별로 객체 클래스가 색상으로 구분된 지도
  * 예: 사람=빨간색, 자전거=파란색, 배경=검은색 등
  * 총 21개 클래스 (배경 + 20개 객체)

핵심 학습 포인트:
1. 이미지 분류 vs 세그멘테이션의 차이
   - 분류: "이 이미지에 고양이가 있다" (전체 이미지 1개 라벨)
   - 세그멘테이션: "이 픽셀은 고양이, 저 픽셀은 배경" (픽셀별 라벨)

2. 전처리 방법의 차이
   - 이미지: 정규화 적용 (학습 안정화)
   - 마스크: 정규화 금지 (클래스 ID 보존), NEAREST 보간법 사용

3. 실제 응용 분야
   - 자율주행: 도로, 차량, 보행자 구분
   - 의료영상: 장기, 종양 영역 분할
   - 배경 제거: 인물과 배경 분리
"""

import torch
import torchvision
from torch.utils.data import DataLoader, dataloader
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt

class VOCSegmentationDataset:
    """PASCAL VOC 2012 세그먼테이션 데이터셋 처리
    
    PASCAL VOC 2012 데이터셋을 세그먼테이션 작업에 맞게 로드하고 전처리하는 클래스
    이미지와 해당하는 세그먼테이션 마스크를 함께 제공함
    """

    def __init__(self, root='./data', train=True, download=True):

        # 
        # 이미지 전처리 : 크기 조정 -> 텐서 변환 -> ImageNet 기준 정규화
        # 
        # transforms.Compose()는 여러 이미지 변환을 순차적으로 적용한다.
        self.transform = transforms.Compose([
            # VOC 데이터셋 이미지들은 크기가 제각각이기 때문에
            # 동일한 크기로 통일
            transforms.Resize((256, 256)),

            # PIL -> Tensor로 변환
            # (height, width, color) -> (color, height, width)
            # 변환 공식 : pixel_value / 255.0
            transforms.ToTensor(),

            # ImageNet의 데이터셋의 RGB 채널별 평균과 표준편차
            # 평균들, 표준편차들
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            # 정규화를 두번 진행해 ImageNet 통계에 맞추고, 전이학습 성능을 높인다.
        ])

        # 
        # 세그멘테이션 마스크 전처리
        # 

        # 마스크 처리가 이미지와 다른 이유:
        # 마스크는 정수이지 픽셀  강도가 아님
        # 0=배경, 1=비행기, 2=자전거, ... 20=TV모니터 (총 21개 클래스)
        # 새로운 값을 생성하면 안된다.
        self.target_transform = transforms.Compose([
            # 마스크 크기 조정
            # 크기가 커질때 가장 가까운 곳의 픽셀 값으로 채운다.
            # 크기가 작아질 때 해당 영역의 가장 첫 번째 픽셀 값으로 영역을 대체한다.
            transforms.Resize((256, 256), interpolation=Image.NEAREST),
            # NEAREST 사용 이유:
            # - 클래스 레이블 값 보존 (0, 1, 2, ..., 20만 유지)
            # - 부드러운 보간(bilinear, bicubic)은 중간값 생성으로 부적절
            # - 예시: 클래스 1(비행기)과 클래스 2(자전거) 경계에서
            #   bilinear → 1.3, 1.7 같은 의미 없는 값 생성
            #   NEAREST → 1 또는 2만 유지 (올바른 클래스 레이블)

            # 마스크를 텐서로 변환
            # 정규화는 하지않는다.
            transforms.ToTensor()
        ])

        self.dataset = torchvision.datasets.VOCSegmentation(
            # 데이터 저장 경로
            root=root,
            # VOC 2012 버전 사용
            year='2012',
            # 훈련/검증 데이터셋 선택
            image_set='train' if train else 'val',
            download=download,
            # 이미지 전처리 함수
            transform=self.transform,
            # 마스크 전처리 함수
            target_transform=self.target_transform
        )

    # 데이터셋의 샘플 수
    def __len__(self):
        return len(self.dataset)
    
    # 해당 인덱스의 데이터 샘플
    def __getitem__(self, idx):
        return self.dataset[idx]

    def get_class_names(self):
        # 21개 클래스의 이름 리스트
        return [
            'background', 'aeroplane', 'bicycle', 'bird', 'boat',
            'bottle', 'bus', 'car', 'cat', 'chair', 'cow',
            'diningtable', 'dog', 'horse', 'motorbike', 'person',
            'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor'
        ]

# 시각화, 이미지 세그멘테이션
def visualize_segmentation_data():
    # 훈련용, 자동 다운로드
    dataset = VOCSegmentationDataset(train=True, download=True)
    
    # DataLoader 생성: 배치 크기 4, 
    # 배치: 딥러닝에서 여러 개의 데이터를 묶어서 한 번에 처리하는 단위다.
    dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

    # 이미지와 마스크 추출
    # dataloader 객체를 이터레이터로 만들어 첫 번째 배치만 가져온다
    images, targets = next(iter(dataloader))

    fir, axes = plt.subplots(2, 4, figsize=(16, 8))

    # 이미지 처리 및 시각화
    for i in range(4):

        # 
        # 정규화된 이미지를 원본으로 복원 (ImageNet 정규화 해제)
        # -> transforms.Normalize() 정규화 해제
        # 

        img = images[i]

        # 정규화 공식의 역연산 1 : 표준편차 곱하기
        # view(3, 1, 1) : 브로드캐스팅으로 (3, 256, 256)과 연산 가능 
        img *= torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)

        # 정규화 공식의 역연삭 2 : 평군 더하기
        img += torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)

        # 픽셀값을 0-1 범위로 클리핑
        img = torch.clamp(img, 0, 1)

        # 채널 순서 변경
        # (color, height, width) -> (height, width, color)
        img = img.permute(1, 2, 0)

        # 
        # 마스크 처리
        # 

        # squeeze : (1, 256, 256) -> (256, 256) 불필요한 채널 차원 제거
        # 차원 중 채널이 1인 차원 제거
        # 시각화를 위해 matplotlib은 numpy 배열을 요구
        target = target[i].squeeze().numpy()

        # 
        # 이미지 표시
        # 

        axes[0, i].imshow(img)
        axes[0, i].set_title(f'Original Image {i+1}')
        axes[0, i].axis('off')  # 축 눈금 제거
        # axis('off'): 픽셀 좌표보다 이미지 내용에 집중하도록
        
        # 7-2. 하단 행: 세그먼테이션 마스크 표시
        axes[1, i].imshow(target, cmap='tab20')
        axes[1, i].set_title(f'Segmentation Mask {i+1}')
        axes[1, i].axis('off')  # 축 눈금 제거
        # cmap='tab20': 
        # - 21개 클래스를 구별 가능한 서로 다른 색상으로 표시
        # - tab20은 20개 구별색 + 기본색으로 21개 클래스 커버
        # - 각 클래스별로 다른 색상으로 영역 구분 명확화
    
    # 8. 레이아웃 조정 및 표시
    plt.tight_layout()  # 서브플롯 간격 자동 조정
    # tight_layout(): 제목과 이미지가 겹치지 않도록 여백 최적화
    
    plt.show()  # 그래프 화면에 표시


# 메인 실행부: 세그먼테이션 데이터 시각화 함수 호출
# Purpose: 스크립트 실행 시 즉시 데이터 시각화를 통해 데이터셋 확인
visualize_segmentation_data()