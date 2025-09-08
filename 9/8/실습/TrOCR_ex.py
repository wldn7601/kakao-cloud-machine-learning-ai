from transformers import TrOCRProcessor, VisionEncoderDecoderModel 
from PIL import Image 
import requests 
import torch
import matplotlib.pyplot as plt 
import numpy as np

class TrOCRSystem:
    def __init__(self, model_name='microsoft/trocr-base-printed'):
        print(f"TrOCR 모델: {model_name}")

        self.processor = TrOCRProcessor.from_pretrained(model_name)
        self.model = VisionEncoderDecoderModel.from_pretrained(model_name)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(self.device)
        
    def extract_text(self, image, return_confidence=False):
        if isinstance(image, np.ndarray):
            image = Image.fromarray(image)
        elif isinstance(image, str):
            if image.startswith(('http://', 'https://')):
                image = Image.open(requests.get(image, stream=True).raw)
            else:
                image = Image.open(image)
        
        pixel_values = self.processor(image, return_tensors='pt').pixel_values
        pixel_values = pixel_values.to(self.device)

        with torch.no_grad():
            if return_confidence:
                outputs = self.model.generate(
                    pixel_values,
                    return_dict_in_generate=True,
                    output_scores=True,
                    max_length=256
                )
                generated_ids = outputs.sequences
                token_scores = outputs.scores
            else:
                generated_ids = self.model.generate(pixel_values)
        generated_text = self.processor.batch_decode(generated_ids, skip_special_tokens=True)[0]

        if return_confidence:
            if token_scores:
                token_probs = []
                for score in token_scores:
                    probs = torch.softmax(score, dim=-1)
                    max_prob = torch.max(probs).item()
                    token_probs.append(max_prob)

                confidence = sum(token_probs) / len(token_probs) if token_probs else 0.0

                print(f"신뢰도 계산 상세:")
                print(f"토큰별 확률: {[f'{p:.3f}' for p in token_probs]}")
                print(f"평균 신뢰도: {confidence:.3f}")
            
            else:
                confidence = 0.5
                print("실제 확률 정보 없음, 기본값 사용")

            return generated_text, confidence
        return generated_text
    
    def batch_extract(self, images):
        results = []

        for i, image in enumerate(images):
            print(f"처리 중: {i+1}/{len(images)}")
            try:
                text = self.extract_text(image)
                results.append(text)
            except Exception as e:
                print(f"이미지 {i+1} 처리 실패: {e}")
                results.append("")
            
        return results
    
    def compare_models(self, image):
        models = {
            'Base Printed': 'microsoft/trocr-base-printed',
            'Base Handwritten': 'microsoft/trocr-base-handwritten',
            'Large Printed': 'microsoft/trocr-large-printed'
        }
        results = {}
        for model_name, model_path in models.items():
            try:
                print(f"{model_name} 모델 테스트 중...")
                # 임시 TrOCR 인스턴스 생성
                temp_processor = TrOCRProcessor.from_pretrained(model_path)
                temp_model = VisionEncoderDecoderModel.from_pretrained(model_path)
                temp_model.to(self.device)

                pixel_values = temp_processor(image, return_tensors='pt').pixel_values.to(self.device)
                
                with torch.no_grad():
                    generated_ids = temp_model.generate(pixel_values)
                text = temp_processor.batch_decode(generated_ids, skip_special_tokens=True)[0]
                results[model_name] = text

                del temp_model, temp_processor
                torch.cuda.empty_cache() if torch.cuda.is_available() else None

            except Exception as e:
                print(f"{model_name} 모델 테스트 실패: {e}")
                results[model_name] = f"Error: {str(e)}"
        
        return results
    

def trocr_basic_demo():
    print("TrOCR 데모 시작")

    ocr = TrOCRSystem('microsoft/trocr-base-printed')
    sample_image = create_sample_trocr_image()
    text = ocr.extract_text(sample_image)
    print(f"인식 결과: '{text}")

    print("\n2. 모델 성능 비교")
    try:
        sample_image = create_sample_trocr_image()
        comparsion = ocr.compare_models(sample_image)

        for model_name, result in comparsion.items():
            print(f"{model_name:20}: '{result}'")

    except Exception as e:
        print(f"모델 비교 스킵: {e}")
    
    print("\n3. 신뢰도 계산 테스트")
    try:
        sample_image = create_sample_trocr_image()
        text, confidence = ocr.extract_text(sample_image, return_confidence=True)
        print(f"인식된 텍스트: '{text}'")
        print(f"신뢰도: {confidence:.1f}")
    except Exception as e:
        print(f"신뢰도 테스트 스킵: {e}")
    return ocr

def create_sample_trocr_image():
    from PIL import Image, ImageDraw, ImageFont

    img = Image.new('RGB', (400,100), 'white')
    draw = ImageDraw.Draw(img)
    try:
        font = ImageFont.truetype('arial.ttf', 24)
    except OSError:
        font = ImageFont.load_default()
    
    text = "TrOCR Demo Text"
    draw.text((50, 30), text, fill='black', font=font)
    img.save('sample_trocr_image.png')
    return img

def trocr_performance_test():
    print("\nTrOCR 성능 테스트")

    ocr = TrOCRSystem()

    test_cases = [
        "Hello World", # 기본 영어
        "The quick brown fox", # 긴 영어 문장
        "Machine Learning", # 기술 용어
        "2024년 한국어 테스트", # 한글 + 숫자
        "Mixed 한글 English 123" # 혼합 텍스트
    ]
    accuracies = []

    for test_text in test_cases:
        print(f"\n테스트 중: '{test_text}'")
        test_image = create_test_image(test_text)

        try:
            predicted = ocr.extract_text(test_image)
        except Exception as e:
            print(f"OCR 실행 실패: {e}")
            predicted = ""
        accuracy = calculate_simple_accuracy(test_text, predicted)
        accuracies.append(accuracy)
        print(f"원본: '{test_text}'")
        print(f"예측: '{predicted}'")
        print(f"정확도: {accuracy}")
    
    avg_accuracy = np.mean(accuracies) if accuracies else 0.0
    print(f"\n전체 평균 정확도: {avg_accuracy:.2f}")

    return avg_accuracy

def create_test_image(text):
    from PIL import Image, ImageDraw, ImageFont

    img_width = len(text) * 20 + 100
    img_height = 60

    img = Image.new('RGB', (img_width, img_height), 'white')
    draw = ImageDraw.Draw(img)

    try:
        font = ImageFont.truetype('arial.ttf',20)
    except OSError:
        font = ImageFont.load_default()

    draw.text((20,20), text, fill='black', font=font)
    return img

def calculate_simple_accuracy(ground_truth, predicted):
    if not ground_truth or not predicted:
        return 0.0
    gt_chars = set(ground_truth.lower().replace(' ',''))
    pred_chars = set(predicted.lower().replace(' ',''))

    if len(gt_chars) == 0:
        return 1.0 if len(pred_chars) == 0 else 0.0
    
    intersection = len(gt_chars.intersection(pred_chars))
    union = len(gt_chars.union(pred_chars))

    return intersection / union if union > 0 else 0.0

try:
    ocr_system = trocr_basic_demo()

    performance = trocr_performance_test()
    print(f"\n모든 테스트 완료. 최종 성능: {performance:.2f}")

except Exception as e:
    print(f"실헹 중 오류: {e}")