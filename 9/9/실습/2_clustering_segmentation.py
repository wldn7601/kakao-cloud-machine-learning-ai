from sklearn.cluster import KMeans
from sklearn.cluster import MeanShift
import cv2
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import euclidean_distances

# 이미지를 군집화로 분할
def clustering_segmentation_demo():
    # BGR 기본 이미지 로드
    image = cv2.imread('image.jpg')
    # rgb 이미지로 변환
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # 해당 이미지는 3차원 배열 (height, width, rgb(3))이므로
    # 2차원으로 바꿔주어 색상 기반 클러스터링이 가능하게 한다.
    pixel_values = image_rgb.reshape(-1, 3)

    # K-Means 와 같은 ML 알고리즘들은 float 타입을 선호한다.
    # float32 타입이 더 안정적이고 정확한 계산이 가능
    pixel_values = np.float32(pixel_values)

    # 
    # 1. K-Means 클러스터링
    # K개의 대표 색상으로 이미지 분할
    # 

    # 5개의 색상으로 군집화
    k = 5

    # 고정 랜덤 시드를 사용하는 이유는 KMeans가 시작할 때 중심점을 랜덤하게 초기화해서
    # 실행마다 같은 결과가 주어지지 않을 수 있기에 고정 랜덤 시드를 사용한다.
    kmeans = KMeans(n_clusters=k, random_state=42)

    # fit으로 학습, predict로 어느 집단에 속하는지 예측
    labels = kmeans.fit_predict(pixel_values)

    # 색상 정보를 얻기 위해서 중심값을 찾는다.
    # 이미지는 0-255 정수니 uint8로 변환
    centers = np.uint8(kmeans.cluster_centers_)

    # 이미 labels는 0, 1, 2, 3, 4의 집단으로 나누어짐
    # 예) [1, 0, 3, 2]
    # centers는 사람이 색상정보를 알기 위해서 작성
    # 색상 정보가 [[갈색], [파란색]] 이라고 하면 
    # 해당 픽셀의 집단으로 색상을 바꾸기 위해 사용
    # (262144, 3) 형태
    segmented_image = centers[labels.flatten()]
    # 해당 코드의 동작원리
    # 예) centers = [[갈색], [파란색], [빨간색]]
    # 예) labels = [1, 0, 2, 1, 2]
    # 예) segmented_image = [[파란색], [갈색], [빨간색], [파란색], [빨간색]]
    # 이렇게 된다. 라벨의 숫자가 같다고 해서 같은 행으로 가는게 아니라
    # 해당 라벨 숫자의 centers의 색상 정보를 가져와 해당 label 인덱스의 픽셀의 행으로 간다.

    # 클러스터링이 끝났기에 원래 이미지 차원으로 변경
    segmented_image = segmented_image.reshape(image_rgb.shape)


    # 
    # 2. Mean Shift 클러스터링
    # 데이터 밀도가 높은 영역을 중심으로 클러스터링
    # 

    # 이미지 크기를 줄여서 계산 속도 향샹
    # image.reshpe = (height, width)
    # cv2.resize(image, (width, height))
    # cv2.resize의 순서가 (너비, 높이) 이므로 1과 0의 순서가 바뀜
    small_image = cv2.resize(image_rgb, (image_rgb.shape[1]//2, image_rgb.shape[0]//2))
    small_pixel_values = small_image.reshape(-1, 3)
    small_pixel_values = np.float32(small_pixel_values)

    # 픽셀 샘플링
    # 최대 10000개 픽셀만 사용
    small_size = min(10000, len(small_pixel_values))
    # sample_pixel_values의 인덱스에서 랜덤하게 
    # small_size만큼 중복을 허용해서 인덱스를 선택해 가져온다.
    sample_indices = np.random.choice(len(small_pixel_values), small_size, replace=True)
    # 가져온 인덱스를 통해 픽셀값을 가져온다.
    sampled_pixels = small_pixel_values[sample_indices]

    # 클러스터링 수행
    # bandwidth는 어느 정도의 거리 범위 내로 하나의 그룹으로 볼거냐
    # RGB 색상 공간에서의 거리
    # 픽셀A = [100, 150, 200]  # 파란색 계열
    # 픽셀B = [110, 160, 210]  # 비슷한 파란색 (거리: 약 17)
    # 픽셀C = [200, 50, 50]    # 빨간색 (거리: 약 200)
    # bandwidth=30이면:
    # 픽셀A와 픽셀B는 같은 클러스터 (거리 17 < 30)
    # 픽셀A와 픽셀C는 다른 클러스터 (거리 200 > 30)
    mean_shift = MeanShift(bandwidth=30)
    
    # 클러스터링 적용
    label_ms_sample = mean_shift.fit_predict(sampled_pixels)

    # 중심점을 얻어 색상 정보 얻기
    centers_ms = np.uint8(mean_shift.cluster_centers_)

    # 유클리드 거리 계산으로 중심점들과 각 픽셀간의 거리 계산
    # 예) [[갈색], [파란색], [빨간색]], [50, 100, 75]
    # 예) 갈색과 거리 계산, 파란색과 거리 계산, 빨간색과 거리 계산
    # 예) 각 픽셀마다 각 색깔과의 거리 계산 걀과를 모두 저장
    # distances = [
    # [거리0→중심0, 거리0→중심1, 거리0→중심2, 거리0→중심3],  # 픽셀 0
    # [거리1→중심0, 거리1→중심1, 거리1→중심2, 거리1→중심3],  # 픽셀 1  
    # [거리2→중심0, 거리2→중심1, 거리2→중심2, 거리2→중심3],  # 픽셀 2
    # ...
    # [거리12499→중심0, 거리12499→중심1, 거리12499→중심2, 거리12499→중심3]  # 픽셀 12499
    # ]
    distances = euclidean_distances(small_pixel_values, centers_ms)

    # 행 방향으로 가장 낮은 거리를 갖는 인덱스를 추출
    #  -> 어떤 집단에 속하는지 알려줌
    labels_ms_full = np.argmin(distances, axis=1)

    # 각 픽셀을 해당 집단의 색깔로 변환
    segmented_image_ms = centers_ms[labels_ms_full]

    # 원래 이미지 차원으로 복구
    segmented_image_ms = segmented_image_ms.reshape(small_image.shape)

    # 원래 이미지 크기로 변경
    segmented_image_ms = cv2.resize(segmented_image_ms, (image_rgb.shape[1], image_rgb.shape[0]))

    # 시각회
    plt.figure(figsize=(15, 5))
        
    plt.subplot(1, 3, 1)
    plt.imshow(image_rgb)
    plt.title('Original Image')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(segmented_image)
    plt.title(f'K-Means (k={k})')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(segmented_image_ms)
    plt.title('Mean Shift (Optimized)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

clustering_segmentation_demo()