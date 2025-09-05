import cv2
import numpy as np
import torch
import re
# from PIL import Image, ImageDraw, ImageFont
import pytesseract


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