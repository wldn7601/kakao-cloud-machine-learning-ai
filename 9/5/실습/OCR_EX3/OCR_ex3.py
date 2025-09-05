import cv2
import numpy as np
import torch
import re
# from PIL import Image, ImageDraw, ImageFont
import pytesseract
import matplotlib.pyplot as plt


class OCRPreprocessor:
    def __init__(self) -> None:
        pass

    # 컬러 이미지를 그레이스케일로 변환
    def convert_to_grayscale(self, image):
        # 이미지 차원이 3일 때
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        return gray

    # 이미지에 이진화 처리를 적용하여 흑백으로 변환
    def apply_threshold(self, image, method='adaptive'):
        gray = self.convert_to_grayscale(image)

        if method == 'simple':
            # 단순 임계값: 127보다 밝으먄 255(흰색), 어두우면 0(검은색)
            _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)
            # _: 임계값 반환
            # thresh: 이진화된 이미지 반환
        
        elif method == 'adaptive':
            # 적응형 임계값: 각 픽셀 주변 영역의 가중 평균을 기준으로 임계값 결정
            # 11x11 영역의 가우시안 가중 평균에서 2를 뺀 값을 임계값으로 결정
            thresh = cv2.adaptiveThreshold(
                # 그레이 스케일 이미지, 최대값, 임계값 방법
                gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                # 이진화 방법, 영역크기, 상수값
                cv2.THRESH_BINARY, 11, 2
            )

        elif method == 'otsu':
            # Otsu 방법: 히스토그램 분석으로 최적의 임계값 자동 선택
            # 히스토그램 분석: 이미지의 픽셀 값 분포를 분석하여 최적의 임계값을 찾는 방법
            
            # 임계값, 이진화된 이미지
            _, thresh = cv2.threshold(
                # 그레이 스케일 이미지, 임계값, 최대값, 이진화 방법
                gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
            )
            
        return thresh
    
    # 이진화된 이미지에서 노이즈 제거
    def remove_noise(self, image):
        # 3x3 구조 요소 생성(모폴로지 연산용)
        # 모폴로지 연산: 이미지의 픽셀 값을 변경하는 연산
        kernel = np.ones((3,3), np.uint8)

        # 열림 연산 적용: 작은 노이즈 점들을 제거
        # 먼저 침식으로 작은 객체를 제거하고, 팽창으로 원래 크기 복원
        # 침식: 이미지의 픽셀 값이 0인 경우, 주변 픽셀값을 0으로 변경
        # 팽창: 이미지의 픽섹 값이 0이 아닌 경우, 주변 픽셀 값을 0이 아닌 값으로 변경
        opening = cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

        # 닫힘 연산 추가: 텍스트 내부의 작은 구멍들을 매움
        # 먼저 팽창으로 구멍을 메우고, 침식으로 원래 크기 복원
        closing = cv2.morphologyEx(opening, cv2.MORPH_CLOSE, kernel)

        # 3x3 영역의 중앙값으로 노이즈 제거(에지 보존 효과)
        #  이미지, 커널 크기, sigmax
        denoised = cv2.GaussianBlur(closing, (3,3), 0)
        return denoised

    # 이미지의 기울기를 보정
    def correct_skew(self, image):
        # Canny 에지 검출: 50-150 임계값으로 윤곽선 추출
        # 그레이스케일 이미지, 최소 임계값, 최대 임계값, 커널 크기
        edges = cv2.Canny(image, 50, 150, apertureSize=3)

        # 허프 변환으로 직선 검출
        # 에지 이미자, 거리 해상도, 각도 해상도, 최소 투표 수
        lines = cv2.HoughLines(edges, 1, np.pi/180, threshold=100)

        if lines is not None:
            # 직선들의 각도 계산
            angles = []
            for rho, theta in lines[:, 0]:
                # 허프 변환에서 theta:
                # 0 -> 수직선, 90 -> 수평선, 180 -> 수직선
                # 이미지 회전에서 각도의 의미:
                # 0 -> 수평선 기준, 90 -> 수직선 기준, 
                # 양수 -> 시계방향 회전, 음수 -> 반시계방향 회전
                # 즉 허프 변환은 수직선을 0도로 보지만, 이미지 회전은 수평선을 0도로 보기 때문에 -90도 보정
                angle = np.degrees(theta) -90
                angles.append(angle)
            
            # 중앙값으로 기울기 보정
            median_angle = np.median(angles)
            
            # 이미지 중심점을 기준으로 회전 변환 행렬 생성
            (h, w) = image.shape[:2]
            center = (w // 2, h // 2)
            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)

            # INTER_CUBIC: 고품질 보간, BORDER_REPLICATE: 가장자리 복제
            # 이미지, 회전 행렬, 출력이미지 크기, 보간법, 테두리 처리
            rotated = cv2.warpAffine(image, M, (w, h), 
                                    flags=cv2.INTER_CUBIC,
                                    borderMode=cv2.BORDER_REPLICATE)
            return rotated
        return image

    # OCR 최적화를 위한 이미지 크기 조정
    def resize_image(self, image, target_height=800):
        h, w = image.shape[:2]

        if h < target_height:
            scale = target_height / h
            new_w = int(w*scale)

            # 
            resized = cv2.resize(image, (new_w, target_height),
                                interpolation=cv2.INTER_CUBIC)

        else:
            resized = image
        return resized

    # 전처리 과정을 단계별로 시각화
    def visualize_preprocessing_steps(self, steps, step_names):
        plt.rc('font', family='Apple SD Gothic Neo')
        # 2행 3열의 서브플롯, (15x10 인치 크기)
        fig, axes = plt.subplots(2, 3, figsize=(15,10))
        
        axes = axes.ravel()

        for i, (step, name) in enumerate(zip(steps, step_names)):
            if i < len(axes):
                if len(step.shape) == 3:
                    axes[i].imshow(cv2.cvtColor(step, cv2.COLOR_BGR2RGB))
                else:
                    axes[i].imshow(step, cmap='gray')
                
                axes[i].set_title(name)
                axes[i].axis('off')
            
        plt.tight_layout()
        plt.show()

    # 전체 이미지 전처리 파이프라인 실행
    def preprocessing_pipeline(self, image, visualize=False):
        steps = []
        step_names = []

        steps.append(image.copy())
        step_names.append('원본 이미지')

        # 1. 그레이스케일 변환
        gray = self.convert_to_grayscale(image)
        steps.append(gray)
        step_names.append('그레이스케일')

        # 2. OCR 최적화를 위한 크기 조정
        resized = self.resize_image(gray)
        steps.append(resized)
        step_names.append('크기 조정')

        # 3. 적응적 이진화 처리
        thresh = self.apply_threshold(resized, method='adaptive')
        steps.append(thresh)
        step_names.append('이진화')

        # 4. 노이즈 제거
        denoised = self.remove_noise(thresh)
        steps.append(denoised)
        step_names.append('노이즈 제거')

        # 5. 기울기 보정
        corrected = self.correct_skew(denoised)
        steps.append(corrected)
        step_names.append('기울기 보정')

        if visualize:
            self.visualize_preprocessing_steps(steps, step_names)

        return corrected

# 노이즈가 있는 샘플 이미지 생성
def create_noisy_sample_image():
    image = np.ones((300, 800, 3), dtype=np.uint8) * 255

    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image, 'Noisy OCR Test Image', (50, 100), font, 1.5, (0, 0, 0), 2)
    cv2.putText(image, 'preprocessing improves accuracy', (50, 150), font, 1, (0,0,0), 2)
    cv2.putText(image, 'Machine Learning & AI', (50,200), font, 1, (0,0,0),2)

    noise = np.random.randint(0, 50, image.shape, dtype=np.uint8)
    noisy_image = cv2.add(image, noise)

    h, w = noisy_image.shape[:2]

    center = (w//2, h//2)
    M = cv2.getRotationMatrix2D(center, 5, 1.0)
    skewed_image = cv2.warpAffine(noisy_image, M, (w,h))

    return skewed_image

# OCR 전처리 실행 함수
def preprocessing_example():
    preprocessor = OCRPreprocessor()

    image = create_noisy_sample_image()

    processed_image = preprocessor.preprocessing_pipeline(image, visualize=True)

    try:
        original_text = pytesseract.image_to_string(image)
        processed_text = pytesseract.image_to_string(processed_image)

        print("전처리 전 OCR 결과:")
        print(repr(original_text))
        print("\n전처리 후 OCR 결과:")
        print(repr(processed_text))
    
    except Exception as e:
        print(f"{e}")
    return processed_image

preprocessing_example()


