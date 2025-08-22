# index, value 형태의 튜플로 반환
fruits = ["사과", "바나나", "체리", "딸기", "오렌지", "블루베리"]
print(list(enumerate(fruits)))
for index, fruit in enumerate(fruits):
    print(f"index: {index}, fruit: {fruit}", end="\t")
print()

# enumerate 실습
colors = ["회색", "노란색", "흰색", "검은색", "회색"]
gray_color_index = [i for i, color in enumerate(colors) if color == "회색"]
print(gray_color_index)
print()

# zip, enumerate 실습
items = ['노트북', '마우스', '키보드', '모니터', '웹캠']
prices = [1200000, 35000, 80000, 250000, 45000]
for index, (item, price) in enumerate(zip(items, prices), 1):
    if price > 50000:
        print(f"상품 번호: {index}, 상품명: {item}, 가격: {price}원")
print()