# 중첩 리스트 컴프리헨션 생성
# 5 X 5 배열(행렬)
rows = cols = 5
matrix = [[row*cols + col for col in range(cols)] for row in range(rows)]
print(matrix)
print()

# 컴프리헨션 실습

# 짝수는 제곱, 홀수는 그대로
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# if만 사용하면 뒤에, if-else를 사용하면 앞에
numbers2 = [number**2 if number%2 == 0 else number for number in numbers ]
print(numbers2)
print()

# 100000원 이상의 제품을 딕셔너리로 만듦
items = ['노트북', '마우스', '키보드', '모니터']
prices = [1200000, 35000, 80000, 250000]
item_dict = {item : price for item, price in zip(items, prices) if price >= 100000}
print(item_dict)
print()

# 5글자 이상의 글자만 넣기
words = ['python', 'is', 'a', 'powerful', 'programming', 'language']
long_words = [word.upper() for word in words if len(word) >= 5]
print(long_words)
print()

# 2차원을 1차원으로 
matrix = [
    [1, 2, 3],
    [4, 5, 6],
    [7, 8, 9]
]
list_matrix = [num for num_list in matrix for num in num_list]
print(list_matrix)
print()

# 재고가 10개 이상인 제품만 딕셔너리에 넣기
inventory = {'사과': 15, '바나나': 8, '딸기': 25, '포도': 12, '오렌지': 5}
new_inventory = {name:num for name, num in inventory.items() if  num>= 10}
print(new_inventory)
print()

