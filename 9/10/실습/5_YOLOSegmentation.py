
from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt

class YOLOSegmentaion:
    """YOLO를 활용한 인스턴스 세그먼테이션"""
    # YOLOSegmentation 클래스: YOLO(You Only Look Once) 모델을 사용하여 인스턴스 세그먼테이션을 수행하는 클래스
    # 인스턴스 세그먼테이션: 객체 검출과 동시에 각 객체의 정확한 경계(마스크)를 픽셀 단위로 분할하는 기술
    # YOLO 사용 이유: 실시간 처리가 가능하고 정확도가 높으며, 사용이 간편함
    
    def __init__(self, model_name='yolov8n-seg.pt'):
        # 모델 생성
        self.model = YOLO(model_name)

    # 예측 및 시각화
    def predict_and_visualize(self, image_path, confidence=0.5):
        # confidence이상 확신할 때만 객체로 인식
        # 호출한 이미지에 대한 추론 실행
        results = self.model(image_path, conf=confidence)
        
        # 결과 시각화
        # results는 리스트 형태로 반환되므로 반복문
        for r in results:
            img = cv2.imread(image_path)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            # 원본 이미지
            axes[0].imshow(img_rgb)
            axes[0].set_title('Original Image')
            axes[0].axis('off')
            
            # 바운딩 박스와 라벨
            img_with_boxes = r.plot()

            axes[1].imshow(img_with_boxes)
            axes[1].set_title('Detection Results')
            axes[1].axis('off')

            # YOLO에서 마스크는 각 픽셀이 어떤 객체에 속하는지 구분해준다.
            # 마스크 수는 YOLO가 인식한 객체의 수이다.
            # 마스크만 표시
            # 세그멘테이션 마스크 정보가 있을 경우
            if r.masks is not None:
                masks = r.masks.data.cpu().numpy()

                # 이미지와 같은 크기의 빈 배열 생성
                combined_mask = np.zeros_like(img_rgb)

                # 검출된 객체별로 마스크 처리
                # mask는 객체를 검출한 부분이지만 정확하게 검출한 것은 아니다
                # 원래 객체보다 더 크게 범위를 잡아서 각 픽셀들의 확률을 계산한다.
                for i, mask in enumerate(masks):
                    # 마스크를 원본 이미지에 맞게 조정
                    # YOLO 출력 이미지와 원본 이미지의 크기가 다를 수 있다.
                    # 마스크는 (width, height) 순서
                    # 해당 객체 (r)에 속하는 모든 픽셀들의 확률을 가지고 있다.
                    # 예)
                    # - mask_resized = array([
                    # -      [0.0, 0.0, 0.0, 0.0, 0.0],  # 배경 영역 = 0
                    # -      [0.0, 0.7, 0.9, 0.8, 0.0],  # 사람의 머리 부분 = 높은 확률
                    # -      [0.0, 0.6, 0.95, 0.9, 0.0], # 사람의 몸통 부분 = 높은 확률  
                    # -      [0.0, 0.0, 0.3, 0.0, 0.0]   # 경계/애매한 부분 = 낮은 확률
                    # - ])
                    mask_resized = cv2.resize(mask, (img_rgb.shape[1], img_rgb.shape[0]))

                    # RGB 색상을 랜덤하게 생성
                    # 예) [123, 14, 200]
                    color = np.random.randint(0, 255, 3)
                    
                    # 해당 
                    colored_mask = np.zeros_like(img_rgb)
                    
                    # 확률이 0.5보다 높은 것만 색상을 칠해 객체로 본다.
                    colored_mask[mask_resized > 0.5] = color

                    # 기존에 있던 마스크는 그냥 놔두고
                    # 새로 추가되는 마스크는 70% 투명도로 추가한다.
                    # 이미지에 여러 마스크를 표시하기 위해 필요
                    combined_mask = cv2.addWeighted(combined_mask, 1, colored_mask, 0.7, 0)

                # 최종 마스크를 원본 이미지에 겹치기
                result_img = cv2.addWeighted(img_rgb, 0.6, combined_mask, 0.4, 0)

                axes[2].imshow(result_img)
                axes[2].set_title('Segmentation Masks')
            else:
                # 마스크가 없는 경우 대체 텍스트 표시
                axes[2].text(0.5, 0.5, 'No masks detected', transform=axes[2].transAxes, ha='center', va='center')
                # transform=axes[2].transAxes: 축 좌표계 사용 (0~1 범위)
                # ha='center', va='center': 수평, 수직 중앙 정렬
                axes[2].set_title('No Segmentation Results')
            
            axes[2].axis('off')
            plt.tight_layout()
            # plt.tight_layout(): 서브플롯 간 간격 자동 조정으로 겹침 방지
            plt.show()

            # 검출된 객체 정보 출력
            # r.boxes : 검출된 객체들의 바운딩 박스
            if r.boxes is not None:
                print(f"검출된 객체 수: {len(r.boxes)}")

                for i, box in enumerate(r.boxes):
                    # 검출된 각 객체에 대한 상세 정보 출력
                    # tensor 타입
                    class_id = int(box.cls[0])
                    # 원래 tensor 타입
                    confidence = float(box.conf[0])

                    # 숫자가 키 : 이름이 밸류
                    class_name = self.model.names[class_id]
                    print(f"객체 {i+1}: {class_name} (신뢰도: {confidence:.2f})")

def yolo_segmentation_demo():
    yolo_seg = YOLOSegmentaion()
    yolo_seg.predict_and_visualize('image.jpg', confidence=0.5)
    # 모델이 인식할 수 있는 모든 객체 클래스의 목록을 출력
    print(f"지원하는 클래스: {list(yolo_seg.model.names.values())}")

yolo_segmentation_demo()

