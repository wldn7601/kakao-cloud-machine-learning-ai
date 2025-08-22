# 문제 1: 짝수만 더하기
# 1부터 100까지의 숫자 중에서 짝수만 골라 합을 계산하는 get_even_sum이라는 함수를 작성하세요. for 반복문을 사용해야 합니다.

def get_even_sum():
    sum = 0
    for num in range(1, 101):
        if num % 2 == 0:
            sum += num
    return sum
print(get_even_sum())
print()

# 문제 2: 리스트 요소 변환
# 주어진 문자열 리스트 fruits의 모든 요소를 대문자로 변환하여 새로운 리스트를 반환하는 convert_to_uppercase 함수를 작성하세요. 
# map() 함수와 lambda식을 사용해야 합니다.

fruits = ['apple', 'banana', 'cherry']

def convert_to_uppercase(fruit_list):
    uppercase_fruits = list(map(lambda fruit: fruit.upper() , fruit_list))
    return uppercase_fruits

uppercase_fruits = convert_to_uppercase(fruits)
print(f"대문자 변환된 과일 리스트: {uppercase_fruits}")
print()

# 문제 3: 사용자 정보 필터링
# 주어진 딕셔너리 users에서 나이가 30세 이상인 사용자만 골라 
# filtered_users라는 새로운 딕셔너리를 반환하는 filter_by_age 함수를 작성하세요. 
# 딕셔너리 컴프리헨션을 사용해야 합니다.

users = {
    '김민지': 25,
    '이서준': 32,
    '박서윤': 28,
    '정우진': 45
}

# 함수를 작성하세요
def filter_by_age(user_dict):
    filtered_users = {name: age for name, age in users.items() if age >= 30}
    return filtered_users

filtered_users = filter_by_age(users)
print(f"30세 이상인 사용자: {filtered_users}")
print()

# 문제 4: 학생 점수 통계 함수
# 주어진 학생 점수 리스트에서 평균, 최고 점수, 최저 점수를 계산하여 딕셔너리로 반환하는 get_score_stats 함수를 작성하세요. 
# 단, 이 함수는 빈 리스트가 입력될 경우 None을 반환해야 합니다.

scores1 = [88, 92, 75, 100, 65]
scores2 = []

def get_score_stats(score_list):
    if not score_list:
        return None
    max_score = max(score_list)
    min_score = min(score_list)
    avg = round(sum(score_list) / len(score_list), 2)
    return {
        "최고 점수" : max_score,
        "최저 점수" : min_score,
        "평균" : avg
    }

# 함수 호출 및 결과 출력
stats1 = get_score_stats(scores1)
print(f"점수 통계 1: {stats1}")
# 예상 출력: {'평균': 84.0, '최고점': 100, '최저점': 65}

stats2 = get_score_stats(scores2)
print(f"점수 통계 2: {stats2}")
# 예상 출력: None
print()

# 문제 5: 리스트 변환 및 필터링
# 주어진 리스트 numbers에서 홀수만 필터링한 후, 각 숫자를 제곱하여 새로운 리스트를 반환하는 코드를 작성하세요.
# 조건:
    # filter() 함수와 lambda식을 사용해야 합니다.
    # map() 함수와 lambda식을 사용해야 합니다.

numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

odd_squared =list(map(lambda num: num**2, filter(lambda x: x % 2 == 1, numbers)))

print(odd_squared)
# 예상 출력: [1, 9, 25, 49, 81]
print()

# 문제 6: 문자열 처리 및 집계
# 주어진 문자열 리스트 data에서 길이가 5 이상인 문자열만 필터링하고, 그 문자열들의 길이 합계를 구하는 코드를 작성하세요.
# 조건:
    # filter() 함수와 lambda식을 사용해야 합니다.
    # reduce() 함수와 lambda식을 사용해야 합니다.

from functools import reduce

data = ["apple", "banana", "kiwi", "grapefruit", "orange", "melon"]

# 여기에 코드를 작성하세요.
# reduce() 첫 번째는 누적값, 두번째는 객체에서 가져온 값, 마지막 0은 누적값의 초기화 값
total_length = reduce(lambda length, str : length + len(str), filter(lambda str: len(str) >= 5, data), 0)

print(total_length)
print()
# 예상 출력: 32 (banana:6 + grapefruit:10 + orange:6 + melon:5)

# 문제 7: 복합 컴프리헨션
# 주어진 두 개의 리스트 fruits와 prices를 사용하여, 가격이 2000원 이상인 과일만 골라 {과일이름: 가격} 형태의 딕셔너리로 만드는 코드를 작성하세요.
# 조건:
    # 딕셔너리 컴프리헨션을 사용해야 합니다.
    # zip() 함수를 사용해야 합니다.

fruits = ['apple', 'banana', 'cherry', 'grape']
prices = [1500, 2500, 3000, 1800]

# 여기에 코드를 작성하세요.
expensive_fruits = {fruit : price for fruit, price in zip(fruits, prices) if price >= 2000}

print(expensive_fruits)
# 예상 출력: {'banana': 2500, 'cherry': 3000}
print()

# 문제 8: 최상 난이도 - 종합 문제
# 주어진 딕셔너리 employees에서 근속연수가 5년 이상이고, 급여가 6000만 원 미만인 직원들만 필터링한 후, 
# 그들의 급여를 10% 인상하여 새로운 딕셔너리로 반환하는 코드를 작성하세요.
# 조건:
    # 딕셔너리 컴프리헨션을 사용해야 합니다.
    # 두 가지 if 조건을 포함해야 합니다.

employees = {
    '김민지': {'years': 7, 'salary': 5500},
    '이서준': {'years': 4, 'salary': 6200},
    '박서윤': {'years': 5, 'salary': 7000},
    '정우진': {'years': 10, 'salary': 5800}
}

# 여기에 코드를 작성하세요.
salary_increased_employees = {name : round(info['salary'] * 1.1, 2) for name, info in employees.items() if info["years"] >= 5 and info["salary"] < 6000}

print(salary_increased_employees)
# 예상 출력: {'김민지': 6050.0, '정우진': 6380.0}


# 아래서 부터 별표 쌓기 
# 가로: row개의 줄
# row개 만큼의 숫자를 입력한 뒤 각 숫자만큼 아래에서 쌓기

def bottom_build_star(numbers):
    max_num = max(numbers)
    for level in range(max_num, 0, -1):
        for num in numbers:
            if num - level >= 0:
                print("*", end="")
            else:
                print(" ", end="")
        print()


col = int(input("가로줄의 개수: "))
numbers = []
for i in range(col):
    numbers.append(int(input(f"{i+1} 번쩨 숫자 입력: ")))
bottom_build_star(numbers)