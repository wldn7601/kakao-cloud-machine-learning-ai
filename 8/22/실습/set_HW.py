
users_info = {}

# 유저 추가
def add_user_info():
    print("\n=== 유저 추가 ===")
    name = input("이름을 입력하세요: ")
    # 이미 존재하는 이름일 때
    if name in users_info:
        print(f"'{name}'은 이미 존재하는 유저입니다.")
        return
    # 존재하지 않는 이름일 때
    # 취미를 3번 입력
    hobbies = []
    print("3개의 취미를 입력하세요")
    for i in range(3):
        hobby = input(f"{i+1}번쩨 취미를 입력하세요: ")
        hobbies.append(hobby)
    users_info[name] = hobbies
    print(f"'{name}' 유저가 추가되었습니다.")

# 유저 삭제
def delete_user_info():
    print("\n=== 유저 삭제 ===")
    name = input("이름을 입력하세요: ")
    # 존재하지 않는 이름일 때
    if name not in users_info:
        print(f"'{name}'은 존재하지 않는 유저입니다.")
        return
    # 존재하는 이름일 때
    del users_info[name]
    print(f"'{name}' 유저가 삭제되었습니다.")

# 유저 검색
def get_user_info():
    print("\n=== 유저 검색 ===")
    name = input("이름을 입력하세요: ")
    # 존재하지 않는 이름일 때
    if name not in users_info:
        print(f"'{name}'은 존재하지 않는 유저입니다.")
        return
    # 존재하는 이름일 때
    hobbies = users_info[name]
    print(f"'{name}'의 취미는 {hobbies}입니다.")

# 모든 유저의 이름과 취미 출력
def get_users_info():
    print("\n === 모든 유저 출력 ===")
    # 등록된 유저가 없을 때
    if not users_info:
        print("현재 등록된 유저가 없습니다.")
        return
    # 등록된 유저가 있을 때
    for name, hobbies in users_info.items():
        print(f"'{name}'의 취미는 {hobbies}입니다.")

# 조회한 이름과 공통 취미가 있는 유저의 이름과 공통 취미 출력
def get_common_hobbies():
    print("\n=== 공통 취미를 갖는 유저 응답 ===")
    user_name = input("이름을 입력하세요: ")
    # 조회한 이름이 없을 때
    if user_name not in users_info:
        print(f"'{user_name}'은 존재하지 않는 유저입니다.")
        return

    # 공통 취미를 가진 사람이 있으면 True, 없으면 False
    has_common_hobby = False
    # 해당 유저의 취미를 집합으로
    user_set_hobbies = set(users_info[user_name])

    print(f"\n### '{user_name}'와(과) 공통 취미를 가진 유저 ###")
    for name, hobbies in users_info.items():
        # 현재 유저면 다시 실행
        if user_name == name:
            continue

        # 유저의 취미를 집합으로
        set_hobbies = set(hobbies)
        common_hobbies = user_set_hobbies & set_hobbies
        # 공통 취미가 없으면 다시 실행
        if not common_hobbies:
            continue

        # 공통 취미가 있으면 실행하는 코드들
        has_common_hobby = True
        print(f"이름: '{name}', 공통 취미: ", end="")
        for common_hobby in common_hobbies:
           print(f"{common_hobby}", end=" ")
        print()

    # 공통 취미를 가진 사람이 없을 때
    if not has_common_hobby:
        print(f"'{user_name}'와(과) 공통 취미를 가진 유저가 없습니다.")

# 공통 취미가 없는 유저의 이름과, 취미를 찾는 함수
def get_not_common_hobbies():
    print("\n=== 공통 취미가 없는 유저 응답 ===")
    user_name = input("이름을 입력하세요: ")
    # 현재 이름이 없을 때
    if user_name not in users_info:
        print(f"'{user_name}'은 존재하지 않는 유저입니다.")
        return
    
    # 공통 취미를 가진 사람이 있으면 False, 없으면 True
    has_not_common_hobby = False
    # 해당 유저의 취미를 집합으로
    user_set_hobbies = set(users_info[user_name])

    print(f"\n### '{user_name}'와(과) 공통 취미 없는 유저 ###")
    for name, hobbies in users_info.items():
        # 조회한 유저면 건너뛰기
        if user_name == name:
            continue

        # 유저의 취미를 집합으로
        set_hobbies = set(hobbies)
        common_hobbies = user_set_hobbies & set_hobbies
        # 공통 취미가 있으면 다시 실행
        if common_hobbies:
            continue

        # 공통 취미가 없으면 실행하는 코드들
        has_not_common_hobby = True
        print(f"이름: '{name}', 공통되지 않는 취미: ",end="")
        for hobby in set_hobbies:
            print(f"{hobby}", end=" ")
        print()
    
    # 공통 취미가 없는 유저가 없을 때
    if not has_not_common_hobby:
        print(f"'{user_name}'와(과) 공통 취미가 없는 유저가 없습니다.")

# 메뉴 함수
def main_menu():
    print("\n메뉴번호: 1.유저추가 | 2.유저삭제 | 3.유저검색 | 4.공통취미를 가진 유저응답 | 5.공통취미가 없는 유저응답 | 6. 모든 유저출력 | 7.종료")
    choice = input("원하는 메뉴를 입력하세요: ")
    return choice

# 실제 실행 함수
while True:
    choice = main_menu()
    # 유저 추가
    if choice == '1':
        add_user_info()
    # 유저 삭제
    elif choice == '2':
        delete_user_info()
    # 유저 검색
    elif choice == '3':
        get_user_info()
    # 공통 취미를 가진 유저 응답
    elif choice == '4':
        get_common_hobbies()
    # 공통 취미가 없는 유저 응답
    elif choice == '5':
        get_not_common_hobbies()
    # 모든 유저 출력
    elif choice == '6':
        get_users_info()
    # 종료
    elif choice == '7':
        print("프로그램을 종료합니다.")
        break
    # 잘못된 메뉴 번호
    else:
        print("잘못된 메뉴 번호입니다.")
