students = []

def add_student():
    student_name = input("이름: ")
    student_score = int(input("점수: "))
    students.append([student_name, student_score])

def remove_student():
    removed_student_name = input("식제할 이름: ")
    for student in students:
        if student[0] == removed_student_name:
            students.remove(student)
            break
def modify_score():
    modified_student_name = input("점수를 수정할 이름: ")
    modified_score = int(input("수정할 점수: "))
    for i, student in enumerate(students):
        if student[0] == modified_student_name:
            students[i][1] = modified_score
            break
def print_students():
    for student in students:
        print(f"이름: {student[0]}, 점수: {student[1]}")
def score_statistics():
    # 캄프리헨션으로 점수만 뽑아오기
    only_scores = [score for name, score in students]
    max_score = max(only_scores)
    min_score = min(only_scores)
    avg = round(sum(only_scores) / len(only_scores), 2)
    print(f"최고점수: {max_score}, 최저점수: {min_score}, 평균점수: {avg}")

# 사용자가 기능을 선택할 수 있는 메뉴 함수
def main_menu():
    print("\n[메뉴] 1.추가 2.삭제 3.수정 4.출력 5.통계 6.종료")
    choice = input("원하는 메뉴 번호를 입력하세요: ")
    print()
    return choice

# 메인 루프
while True:
    menu_choice = main_menu()
    
    if menu_choice == '1':
        add_student()
    elif menu_choice == '2':
        remove_student
    elif menu_choice == '3':
        modify_score()
    elif menu_choice == '4':
        print_students()
    elif menu_choice == '5':
        score_statistics()
    elif menu_choice == '6':
        print("프로그램을 종료합니다.")
        break
    else:
        print("잘못된 메뉴 번호입니다. 다시 입력해 주세요.")