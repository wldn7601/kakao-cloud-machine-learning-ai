def spiral_matrix(n):
    # n x n 크기의 2차원 리스트 초기화 (0으로 채움)
    matrix = [[0] * n for _ in range(n)]
    
    count = 1
    top = 0 
    left = 0
    right = n - 1
    bottom = n - 1

    while count <= n**2 :
        # (top) 왼 -> 오
        for i in range(left, right + 1):
            matrix[top][i] = count
            count += 1
        top += 1

        # (right) 위 -> 아래
        for i in range(top, bottom + 1):
            matrix[i][right] = count
            count += 1
        right -= 1

        # (bottom) 오 -> 왼
        for i in range(right, left - 1, -1):
            matrix[bottom][i] = count
            count += 1
        bottom -= 1

        # (left) 아래 -> 위
        for i in range(bottom, top - 1, -1):
            matrix[i][left] = count
            count += 1
        left += 1

    return matrix


# 실행 예시
while True:
    N = input("N x N 배열의 N을 입력: ")
    if N.isdecimal() and N > '0':
        N = int(N)
        break
    print("양수를 입력하세요")

result = spiral_matrix(N)

for row in result:
    print(row)
