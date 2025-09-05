from matplotlib import image
import pytesseract
import cv2
import numpy as np 
from PIL import Image 
import matplotlib.pyplot as plt
import pandas as pd
import re

# tesseract ocr 클래스
class TesseractOCR:
    def __init__(self) -> None:
        pass

    # 사용가능한 언러 목록
    def get_available_languages(self):
        lang = pytesseract.get_languages()
        return lang
    
    # 바운딩 박스 포함 ocr
    def ocr_with_bbox(self, image, lang='eng'):
        data = pytesseract.image_to_data(image, lang=lang, output_type=pytesseract.Output.DICT)
        
        # 바운딩 박스 그리기
        
        result_image = image.copy()

        for i in range(len(data['text'])):
            # 신뢰도 60% 초과
            if int(data['conf'][i]) > 60:
                x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
                text = data['text'][i]

                if text.strip():
                    # 바운딩 박스: (x, y, w, h) 좌표, 색상, 두께
                    cv2.rectangle(result_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(result_image, text, (x, y-10), 
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0),2)
        return result_image, data

# 문서 ocr 데모 클래스
class DocumentOCRDemo:
    def __init__(self) -> None:
        self.ocr = TesseractOCR()

    # 영수증 이미지 처리
    def process_receipt_image(self, image_path):
        image = cv2.imread(image_path)
        if image is None:
            # 영수증 생성
            image = self.create_smaple_receipt()

        # 전처리
        processed = self.preprocess_receipt(image)
        # ocr 실행(숫자, 문자, 특수문자 포함)
        config = r'--oem 3 --psm 4 -c tessedit_char_whitelist=0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz., : -$'
        text = pytesseract.image_to_string(processed, config=config)

        # 영수증 정보 파싱
        parsed_info = self.parse_receipt_info(text)

        return text, parsed_info, processed

    # 영수증 전처리
    def preprocess_receipt(self, image):
        # 그레이 스케일 변환
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        
        # 대비 향상
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        enhanced = clahe.apply(gray)

        # 이진화
        _, thresh = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

        return thresh
    
    # 영수증 정보 파싱
    def parse_receipt_info(self, text):
        lines = text.split('\n')
        info = {
            'items':[],
            'total':None,
            'date':None,
            'store_name':None
        }

        for line in lines:
            line = line.strip()
            if not line:
                continue
        price_pattern = r'\$?\d+\.\d{2}'
        if re.search(price_pattern, line):
            if 'total' in line.lower() or 'sum' in line.lower():
                prices = re.findall(price_pattern, line)
                if prices:
                    info['total'] = prices[-1]
            else:
                info['total'].append(line)
        
        date_pattern = r'\d{1,2}/\d{1,2}/\d{4}'
        if re.search(date_pattern, line):
            info['date'] = line
        return info


    # 샘플 영수증 이미지 생성
    def create_smaple_receipt(self):
        image = np.ones((600, 400, 3), dtype=np.uint8) * 255

        font = cv2.FONT_HERSHEY_SIMPLEX
        texts = [
            ('GROCERY STORE', (100, 50), 0.8, 2),
            ('Date: 01/15/2024', (50, 100), 0.6, 1),
            ('Item 1        $5.99', (50, 150), 0.6,1),
            ('Item2         $3.50',(50, 180), 0.6,1),
            ('Item3         $12.25',(50, 210), 0.6, 1),
            ('Tax           $1.73', (50, 260), 0.6, 1),
            ('Total         $23.47', (50, 310), 0.7, 2),
            ('Thank you!', (120, 360), 0.6, 1)
        ]

        for text, pos, scale, thickness in texts :
            cv2.putText(image, text, pos, font, scale, (0,0,0),thickness)
        
        return image

# tesseract oct 실습
def tesseract_practice():
    print("=== tesseract OCR 실습 ===\n")

    print("1. 기본 OCR 테스트")
    ocr = TesseractOCR()

    print("지원 언어:", ocr.get_available_languages()[0:10])

    print("\n2. 영수증 OCR 데모")
    demo = DocumentOCRDemo()
    receipt_text, parsed_info, processed_image = demo.process_receipt_image(None)

    print("영수증 테스트")
    print(receipt_text)
    print("\n파생된 정보:")
    for key, value in parsed_info.items():
        print(f"{key}: {value}")
    
    sample_image = demo.create_smaple_receipt()

    return sample_image, processed_image

sample_image, processed_image = tesseract_practice()

plt.rc('font', family='Apple SD Gothic Neo')
plt.figure(figsize=(15,5))

plt.subplot(1, 3, 1)
plt.imshow(cv2.cvtColor(sample_image, cv2.COLOR_BGR2RGB))
plt.title('원본 영수증')
plt.axis('off')

plt.subplot(1, 3, 2)
plt.imshow(processed_image, cmap='gray')
plt.title('전처리된 이미지')
plt.axis('off')

ocr = TesseractOCR()
bbox_image, _ = ocr.ocr_with_bbox(sample_image)

plt.subplot(1, 3, 3)
plt.imshow(cv2.cvtColor(bbox_image, cv2.COLOR_BGR2RGB))
plt.title('OCR 결과(바운딩 박스)')
plt.axis('off')

plt.tight_layout()
plt.show()


