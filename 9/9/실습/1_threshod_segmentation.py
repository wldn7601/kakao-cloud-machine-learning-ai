import cv2
import numpy as np
import matplotlib.pyplot as plt

# 단순 임계값 기반 이미지 세그멘테이션
def threshold_segmentation_demo():
    # 이미지 로드
    # 0은 그레이스케일로 로드란 의미
    image = cv2.imread('image.jpg', 0)

    # ret: 실제로 사용된 threshold 값 (여기서는 127.0)
    # binary: 이진화된 이미지 배열
    ret, binary = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
    ret2, otsu = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    adaptive = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY, 11, 2)
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 4, 1)
    plt.imshow(image, cmap='gray')
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 4, 2)
    plt.imshow(binary, cmap='gray')
    plt.title('Binary Threshold')
    plt.axis('off')
    
    plt.subplot(1, 4, 3)
    plt.imshow(otsu, cmap='gray')
    plt.title("Otsu's Method")
    plt.axis('off')
    
    plt.subplot(1, 4, 4)
    plt.imshow(adaptive, cmap='gray')
    plt.title('Adaptive Threshold')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    return binary, otsu, adaptive

threshold_segmentation_demo()