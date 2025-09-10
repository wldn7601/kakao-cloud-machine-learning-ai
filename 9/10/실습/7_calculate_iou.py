import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 두 바운딩 박스의 IoU 계산
def calculate_iou(box1, box2):
    x1_inter = max(box1[0], box2[0])
    y1_inter = max(box1[1], box2[1])
    x2_inter = min(box1[2], box2[2])
    y2_inter = min(box1[3], box2[3])

    # 교집합 영역 계산
    if x2_inter > x1_inter and y2_inter > y1_inter:
        intersection = (x2_inter - x1_inter) * (y2_inter - y1_inter)
    else:
        intersection = 0
    
    # 각 박스의 영역 계산
    area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
    area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
    
    # 합집합 영역 계산
    union = area1 + area2 - intersection
    
    # IoU 계산
    iou = intersection / union if union > 0 else 0
    
    return iou

def visualize_iou():
    plt.rc('font', family='Apple SD Gothic Neo')
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 다양한 IoU 케이스
    cases = [
        {'box1': [50, 50, 150, 150], 'box2': [100, 100, 200, 200], 'title': 'IoU = 0.14'},
        {'box1': [50, 50, 150, 150], 'box2': [75, 75, 175, 175], 'title': 'IoU = 0.36'},
        {'box1': [50, 50, 150, 150], 'box2': [60, 60, 140, 140], 'title': 'IoU = 0.64'}
    ]

    # 각 케이스 마다
    for i, case in enumerate(cases):
        box1, box2 = case['box1'], case['box2']
        
        # 빈 이미지
        img = np.ones((250, 250, 3)) * 0.9
        # 0.9 : 배경색
        axes[i].imshow(img)
        
        # 첫 번째 박스 (빨간색)
        # (x, y) : 왼쪽 꼭지점의 시작점, 박스의 너비, 높이
        rect1 = patches.Rectangle((box1[0], box1[1]), box1[2]-box1[0], box1[3]-box1[1],
                                linewidth=2, edgecolor='red', facecolor='red', alpha=0.3)
        # patches.Rectangle: 사각형 박스 생성
        # (box1[0], box1[1]): 박스의 좌상단 좌표
        # box1[2]-box1[0], box1[3]-box1[1]: 박스의 너비와 높이
        # linewidth=2: 선 두께
        # edgecolor='red': 테두리 색상
        # facecolor='red': 내부 색상
        # alpha=0.3: 투명도
        axes[i].add_patch(rect1)
        
        # 두 번째 박스 (파란색)
        # (x, y) : 왼쪽 꼭지점의 시작점, 박스의 너비, 높이
        rect2 = patches.Rectangle((box2[0], box2[1]), box2[2]-box2[0], box2[3]-box2[1],
                                linewidth=2, edgecolor='blue', facecolor='blue', alpha=0.3)
        axes[i].add_patch(rect2)
        
        # IoU 계산 및 표시
        iou = calculate_iou(box1, box2)
        # axes[i].set_title(f'{case["title"]} (실제: {iou:.2f})')
        # axes[i].set_xlim(0, 250)
        # axes[i].set_ylim(250, 0)
        # axes[i].axis('off')
        # iou text로 추가
        axes[i].text(10, 20, f'IoU: {iou:.2f}', fontsize=12, color='black')

    plt.tight_layout()
    plt.show()

visualize_iou()
    