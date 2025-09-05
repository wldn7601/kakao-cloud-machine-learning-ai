import cv2
import numpy as np
import torch
import re
# from PIL import Image, ImageDraw, ImageFont
import pytesseract
import matplotlib.pyplot as plt

def create_sample_image():
    # 흰색 배경 이미지 생성(높이 200, 너비 600, 3채널: RGB)
    image = np.ones((200, 600, 3), dtype=np.uint8) * 255
    # OpenCV를 사용해 이미지에 텍스트 추가
    font = cv2.FONT_HERSHEY_SIMPLEX 
    #                               시작 좌표(x, y)    폰트,  크기, 생상(R,G,B), 두께
    cv2.putText(image, 'Hello OCR World', (50, 100), font, 2, (0, 0, 0), 3)
    cv2.putText(image, 'This is a test image', (50, 150), font, 1, (0, 0, 0), 2)

    cv2.imwrite('sample_text.jpg', image)
    return image

def basic_ocr_example():
    # 실제 이미지 경로
    image_path = 'sample_text.jpg'
    # 이미지 읽기(BGR 형식)
    image = cv2.imread(image_path)

    if image is None:
        print("이미지를 찾을 수 없습니다.")
        # 샘플 이미지 생성
        image = create_sample_image()
    
    # 이미지에서 텍스트를 추출하여 문자열로 변환
    text = pytesseract.image_to_string(image, lang='eng')
    print("인식된 텍스트: ")
    print(text)

    return text, image


text, image = basic_ocr_example()

# 고품질 샘플 이미지 
def create_high_quality_sample():
    # 흰색 배경
    image = np.ones((100, 400, 3), dtype=np.uint8) * 255
    font = cv2.FONT_HERSHEY_SIMPLEX

    cv2.putText(image, 'Perect OCT Text', (20, 60), font, 1.2, (0, 0, 0), 2)
    cv2.imwrite('hihg_quality_sample.jpg', image)
    return image

# 중간 품질 샘플 이미지
def create_medium_quality_sample():
    image = np.ones((100, 400, 3), dtype=np.uint8) * 255

    # 노이즈 
    #                 평균: 0, 표준편차: 15인 정규분포
    noise = np.random.normal(0, 15, image.shape).astype(np.uint8)

    image = cv2.add(image, noise)

    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, 'Noisy OCR Text', (20, 60), font, 1.2, (0, 0, 0), 2)
    cv2.imwrite('medium_quality_sample.jpg', image)

    return image

# 저품질 샘플 이미지
def create_low_quality_sample():
    image = np.ones((100, 400, 3), dtype=np.uint8) * 255

    noise = np.random.normal(0, 30, image.shape).astype(np.uint8)
    image = cv2.add(image, noise)
    font = cv2.FONT_HERSHEY_SIMPLEX

    # (50, 50, 50): 회색 텍스트로 대비 감소
    cv2.putText(image, 'Blurry OCR Text', (20, 60), font, 1.2, (50, 50, 50), 1)

    # 이미지 회전으로 기울기 추가
    rows, cols = image.shape[:2]

    # 회전 변환 병렬 
    # (회전 중심점),  회전 각도, 크기 배율
    M = cv2.getRotationMatrix2D((cols/2, rows/2), 5, 1)
    # 변환 적용
    # OpenCV는 너비, 높이 순서로 인자를 받는다.
    # (x, y) = (cols, rows)
    image = cv2.warpAffine(image, M, (cols, rows))
    
    # 블러 추가
    # 이미지, 커널 크기(3 x 3): x/y 방향, 표준편차
    image = cv2.GaussianBlur(image, (3, 3), 0)
    cv2.imwrite('low_quality_sample.jpg', image)
    return image


def ocr_quality_demo():

    # 다양한 품질의 샘플 비교
    samples = {
        'high_quality': create_high_quality_sample(),
        "medium_quality": create_medium_quality_sample(),
        "low_quality": create_low_quality_sample()
    }

    results = {}

    for quality, image in samples.items():
        # 각 이미지에 대해 ocr 실행
        text = pytesseract.image_to_string(image, lang='eng')
        # OCT 결과의 상세 정보를 딕셔너리로 반환    
        data = pytesseract.image_to_data(image, output_type=pytesseract.Output.DICT)
        print(data)

        # 신뢰도 추출
        # 유효한 텍스트 영역민 필터링
        confidence = [int(conf) for conf in data['conf'] if int(conf) > 0]
        avg_confidence = np.mean(confidence) if confidence else 0

        # 결괴 저장
        results[quality] = {
            'text': text.strip(),
            'confidence': avg_confidence,
            'image': image
        }

        print(f"{quality.upper()}")
        print(f"텍스트: {text.strip()}")
        print(f"평균 신뢰도: {avg_confidence:.2f}%")

    return results


qulaity_results = ocr_quality_demo()

# 결과 시각화
plt.figure(figsize=(15, 20))
plt.rc('font', family='Apple SD Gothic Neo')
plt.subplot(2, 3, 1)

plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
plt.title('기본 OCR 예제')
plt.axis('off')

for i, (quality, result) in enumerate(qulaity_results.items(), 2):
    plt.subplot(2, 3, i)

    plt.imshow(cv2.cvtColor(result['image'], cv2.COLOR_BGR2RGB))

    plt.title(f"{quality.title()}\n신뢰도: {result['confidence']:.2f}%")
    plt.axis('off')

plt.tight_layout()
plt.show()