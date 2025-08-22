# 이름을 키로, 전화번호, 이메일, 주소를 값으로 저장
# 연락처 추가, 삭제, 수정, 보기 기능
# 중첩 딕셔너리로 전화번호, 이메일, 주소도 키:값 형태로

users_info = {}

# 유저 추가
def add_user():
    user_name = input("추가할 유저 이름: ")
    # 사전에 유저가 없는 경우
    if user_name not in users_info:
        user_tel = input("추가할 유저 전화번호: ")
        user_email = input("추가할 유저 이메일: ")
        user_address = input("추가할 유저 주소: ")
        users_info[user_name] = {
            "전화번호" : user_tel,
            "이메일" : user_email,
            "주소" : user_address
        }
        print(f"'{user_name}' 유저가 추가되었습니다.")
    # 이미 유저가 있는 경우
    else:
        print(f"'{user_name}' 유저가 이미 있습니다.")

# 유저 삭제
def remove_user():
    removed_user_name = input("삭제할 유저: ")
    # 삭제할 유저가 있는 경우
    if removed_user_name in users_info:
        del users_info[removed_user_name]
        print(f"'{removed_user_name}' 유저 정보가 삭제되었습니다.")
    # 삭제할 유저가 없는 겅우
    else:
        print(f"'{removed_user_name}' 유저를 찾을 수 없습니다.")

# 유저 정보 수정
def update_user():
    user_name = input("수정할 유저 이름: ")
    # 유저가 있는 경우
    if user_name in users_info:
        user_tel = input("수정할 유저 전화번호: ")
        user_email = input("수정할 유저 이메일: ")
        user_address = input("수정할 유저 주소: ")
        users_info.update({
            user_name : {
                "전화번호" : user_tel,
                "이메일" : user_email,
                "주소" : user_address
            }
        })
        print(f"'{user_name}' 유저 정보가 수정되었습니다.")
    # 유저가 없는 경우
    else:
        print(f"'{user_name}' 유저를 찾을 수 없습니다.")

# 전체 유저 출력
def print_users_info():
    if not users_info:
        print("\n===== 현재 등록된 유저가 없습니다. =====")
        return
    print("\n===== 전체 유저 정보 =====")
    for user_name, user_info in users_info.items():
        print(f"### 이름: {user_name} ###")
        for key, value in user_info.items():
            print(f"{key}: {value}")
        print("-------------------------------------")
# 사용자가 기능을 선택할 수 있는 메뉴 함수
def main_menu():
    print("\n[메뉴] 1.추가 2.삭제 3.수정 4.출력 5.종료")
    choice = input("원하는 메뉴 번호를 입력하세요: ")
    print()
    return choice

# 메인 루프
while True:
    menu_choice = main_menu()
    
    if menu_choice == '1':
        add_user()
    elif menu_choice == '2':
        remove_user()
    elif menu_choice == '3':
        update_user()
    elif menu_choice == '4':
        print_users_info()
    elif menu_choice == '5':
        print("프로그램을 종료합니다.")
        break
    else:
        print("잘못된 메뉴 번호입니다. 다시 입력해 주세요.")