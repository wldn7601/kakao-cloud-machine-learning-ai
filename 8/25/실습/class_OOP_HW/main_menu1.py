import library_class
import member_class

# 
# 도서 제거할 때 회원이 대출 중이면 곤란
# 

def main_menu():
    """사용자에게 메뉴를 보여주고 입력에 따라 함수를 호출합니다."""
    my_library = library_class.Library()
    

    print("--- 시스템 시작 ---")
    print("-" * 20)

    while True:
        print("\n[메뉴] 1.도서추가 2.도서삭제 3.도서검색 4.도서대출 5.도서반납 6.전체도서 7.회원등록 8.전체회원 9.회원탈퇴 10.대출현황 11.종료")
        choice = input("원하는 메뉴 번호를 입력하세요: ")

        # 도서 추가
        if choice == '1':
            print("\n### 도서 추가 ###")
            title = input("도서 제목: ")
            author = input("도서 저자: ")
            isbn = input("도서 ISBN: ")
            year = input("출판 연도: ")
            my_library.add_book(title, author, isbn, year)
        
        # 도서 삭제
        elif choice == '2':
            print("\n### 도서 삭제 ###")
            isbn = input("삭제할 도서의 ISBN: ")
            my_library.remove_book(isbn)
        
        # 도서 검색
        elif choice == '3':
            print("\n### 도서 검색 ###")
            by = input("검색 기준 (title, author, isbn): ")
            query = input("검색할 키워드: ")
            results = my_library.search_book(query, by)
            print("--- 검색 결과 ---")
            if results:
                for book_obj in results:
                    print(book_obj)
            else:
                print("검색 결과가 없습니다.")
        
        # 도서 대출
        elif choice == '4':
            print("\n### 도서 대출 ###")
            member_id = input(f"회원 ID: ")
            isbn = input("대출할 도서의 ISBN: ")
            my_library.lend_book(member_id, isbn)

        # 도서 반납
        elif choice == '5':
            print("\n### 도서 반납 ###")
            member_id = input(f"회원 ID: ")
            isbn = input("반납할 도서의 ISBN: ")
            my_library.return_book(member_id, isbn)

        # 전체 도서 목록
        elif choice == '6':
            print("\n### 도서관 전체 도서 목록 ###")
            my_library.display_all_books()

        # 회원 등록
        elif choice == '7':
            print("\n### 회원 등록 ###")
            member_name = input("회원 이름: ")
            new_member = my_library.register_member(member_name)
            
        # 전체 회원 목록
        elif choice == '8':
            print("\n### 전체 회원 목록 ###")
            my_library.display_all_members()

        # 회원 탈퇴
        elif choice == '9':
            print("\n### 회원 탈퇴 ###")
            member_id = input(f"회원 ID: ")
            my_library.remove_member(member_id)

        # 대출 현황
        elif choice == '10':
            print("\n### 대출 현황 ###")
            member_id = input(f"회원 ID: ")
            borrowed_books = my_library.get_member_borrowed_books(member_id)
            print("--- 대출 현황 ---")
            if borrowed_books:
                for book_obj in borrowed_books:
                    print(book_obj)
            else:
                print("대출한 도서가 없습니다.")

        # 종료
        elif choice == '11':
            print("프로그램을 종료합니다.")
            break
        
        # 잘못된 메뉴 번호
        else:
            print("잘못된 메뉴 번호입니다. 다시 입력해 주세요.")
            
if __name__ == "__main__":
    main_menu()