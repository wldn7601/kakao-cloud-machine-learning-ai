import torch
from torch.cpu import is_available
from torchvision.models.detection import maskrcnn_resnet50_fpn
from torchvision.transforms import functional as F
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

class MaskRCNNPredictor:
    """
    Mask R-CNN을 사용한 인스턴스 세그먼테이션 클래스
    - 사전 훈련된 Mask R-CNN 모델을 사용하여 객체 탐지와 세그먼테이션을 동시에 수행
    - COCO 데이터셋으로 훈련된 모델을 사용하여 80개 클래스의 객체를 인식
    """

    def __init__(self, device='cuda' if torch.cuda.is_available() else 'cpu'):
        # cuda 사용이 가능하면 cuda로 아니먄 cpu로
        self.device = device

        # 사전 훈련된 Mask R-CNN 모델
        self.model = maskrcnn_resnet50_fpn(pretrained=True)
        # 모델을 지정된 디바이스로 이동
        self.model.to(device)
        # 평가 모드
        self.model.eval()

        # COCO 클래스 이름
        self.class_names = [
            '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
            'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
            'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
            'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A',
            'N/A', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard',
            'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
            'surfboard', 'tennis racket', 'bottle', 'N/A', 'wine glass', 'cup', 'fork',
            'knife', 'spoon', 'bowl', 'banana', 'apple', 'sandwich', 'orange',
            'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
            'potted plant', 'bed', 'N/A', 'dining table', 'N/A', 'N/A', 'toilet',
            'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
            'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
            'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
        ]

    # 예측
    def predict(self, image_path, confidence_threshold=0.5):
        # 이미지 PIL로 로드
        image = Image.open(image_path).convert('RGB')

        # to_tensor : [0, 255] 범위의 이미지를 [0, 1] 범위의 텐서로 변환
        # unsqueeze(0) : 배치 차원 추가 (1, color, height, width)
        # 지정된 디바이스로 이동
        image_tensor = F.to_tensor(image).unsqueeze(0).to(self.device)

        # 모델 추론 수행
        # 추론 시에는 그래디언트가 필요 없다.
        with torch.no_grad():
            # 리스트 형태, 각 이미지에 대한 예측은 딕셔너리 형태
            predictions = self.model(image_tensor)
        
        # 하나의 이미지만 처리
        pred = predictions[0]

        # 신뢰도가 임계값 이상인 것만 선택
        # keep_idx는 불리언 텐서
        keep_idx = pred['scores'] > confidence_threshold

        # keep_idx는 불리언 텐서인데 인덱스 역할을 하는거 같이 보인다.
        # 이를 불리언 인덱싱이라고 한다. 해당 배열, 텐서들은 keep_idx와 크기가 같아야한다.
        # keep_idx의 값이 True면 포함, False면 제외한다.
        boxes = pred['boxes'][keep_idx].cpu().numpy()      # 바운딩 박스 좌표
        labels = pred['labels'][keep_idx].cpu().numpy()    # 클래스 레이블
        scores = pred['scores'][keep_idx].cpu().numpy()    # 신뢰도 점수
        masks = pred['masks'][keep_idx].cpu().numpy()      # 세그먼테이션 마스크

        return image, boxes, labels, scores, masks

    # 결과 시각화
    def visualize_results(self, image, boxes, labels, scores, masks):
        # 1x2 서브플롯 생성 (좌: 객체 탐지, 우: 세그먼테이션)
        # - figsize=(15, 7): 충분한 크기로 설정하여 세부사항 확인 가능
        fig, axes = plt.subplots(1, 2, figsize=(15, 7))
        
        # 원본 이미지와 바운딩 박스 표시
        axes[0].imshow(image)
        axes[0].set_title('Object Detection')  # 객체 탐지 결과
        axes[0].axis('off')  # 축 눈금 제거 (이미지에 집중)

        # 각 탐지된 객체에 대해 바운딩 박스와 레이블 그리기
        for i, (box, label, score) in enumerate(zip(boxes, labels, scores)):
            x1, y1, x2, y2 = box  # 박스 좌표 (좌상단, 우하단)
            
            # 사각형 바운딩 박스 생성
            # - Rectangle((x1, y1), width, height): 좌상단 좌표와 크기 지정
            # - linewidth=2: 선 두께, edgecolor='red': 테두리 색상
            # - facecolor='none': 내부 투명 (박스만 표시)
            rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='red', facecolor='none')
            # - 박스 테두리 두께 2, 색상 빨강, 내부 투명
            axes[0].add_patch(rect)
            # add_patch(): 박스를 이미지 위에 추가
            
            # 클래스명과 신뢰도 텍스트 표시
            # - 박스 위쪽에 텍스트 배치 (y1-5)
            # - 클래스명: 레이블 인덱스를 사용하여 실제 이름 표시
            # - 신뢰도: 소수점 둘째 자리까지 표시
            class_name = self.class_names[label]
            axes[0].text(x1, y1-5, f'{class_name}: {score:.2f}', 
                        color='red', fontsize=10, weight='bold')
        
        # 마스크 오버레이 표시
        axes[1].imshow(image)
        axes[1].set_title('Instance Segmentation')  # 인스턴스 세그먼테이션 결과
        axes[1].axis('off')

        # 각 마스크를 다른 색상으로 표시하기 위한 컬러맵 생성
        # - Set1 컬러맵 사용: 구별하기 쉬운 색상들
        # - 마스크 개수만큼 색상 배열 생성
        colors = plt.cm.Set1(np.linspace(0, 1, len(masks)))
        # 매개변수 설명
        # - 0: 첫 번째 색상
        # - 1: 마지막 색상
        # - len(masks): 마스크 개수

        for i, mask in enumerate(masks):
            # 마스크를 RGBA 형태로 변환 = 4채널
            # mask.shape[1:] : (height, width)
            # * : 언팩킹
            mask_colored = np.zeros((*mask.shape[1:], 4))

            # 모든 높이, 너비에 대해
            # RGB의 값 (0, 1, 2)만 colors에서 색상 적용
            mask_colored[:, :, :3] = colors[i][:3]

            # 투명도 적용
            mask_colored[:, :, 3] = mask[0] * 0.7

            axes[1].imshow(mask_colored)

        # 레이아웃 최적화 및 표시
        plt.tight_layout()  # 서브플롯 간격 자동 조정
        plt.show()          # 그래프 표시

# 사용 예제
def demo_maskrcnn():
    """
    Mask R-CNN 데모 함수
    """
    
    # MaskRCNNPredictor 인스턴스 생성
    # - 자동으로 GPU/CPU 선택
    # - 사전 훈련된 모델 로드
    predictor = MaskRCNNPredictor()
    
    # 샘플 이미지로 예측 (실제로는 존재하는 이미지 경로 사용)
    # 실제 사용 시 아래 주석을 해제하고 올바른 이미지 경로 지정
    image, boxes, labels, scores, masks = predictor.predict('living_room.jpg')
    predictor.visualize_results(image, boxes, labels, scores, masks)
    
    # 모델 준비 완료 메시지
    print("Mask R-CNN 모델이 준비되었습니다.")
    print(f"지원하는 클래스 수: {len(predictor.class_names)}")
    print(f"사용 디바이스: {predictor.device}")
    
    # 지원하는 주요 클래스들 출력 (처음 20개)
    print("\n지원하는 주요 클래스들:")
    for i, class_name in enumerate(predictor.class_names[1:21]):  # 배경 제외, 처음 20개
        print(f"  {i+1}: {class_name}")

demo_maskrcnn()  # 함수 호출 (필요시 주석 해제)