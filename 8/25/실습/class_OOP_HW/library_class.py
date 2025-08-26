import book_class
import member_class

class Library:
    """도서와 회원 컬렉션을 관리하고 대출/반납 기능을 제공하는 클래스입니다."""
    def __init__(self) -> None:
        # 비공개 정보
        # 전체 도서 목록
        self._books = {}
        # 전체 회원 목록
        self._members = {}

    def add_book(self, title, author, isbn, year):
        """도서 추가"""
        # 도서는 isbn으로 구분
        # 이미 도서가 있을 때
        if isbn in self._books:
            print(f"'{title}' (ISBN: {isbn}) 도서는 이미 등록되어 있습니다.")
            return
        # 도서가 없을 때 추가
        new_book = book_class.Book(title, author, isbn, year)
        # self._books 딕셔너리. 키: isbn의 값을, 값: 도서 인스턴스
        self._books[isbn] = new_book
        print(f"'{title}' (ISBN: {isbn}) 도서가 성공적으로 추가되었습니다.")
    
    def remove_book(self, isbn):
        """도서 삭제"""
        # isbn 값에 해당하는 도서가 없을 때
        if isbn not in self._books:
            print(f"ISBN '{isbn}'에 해당하는 도서를 찾을 수 없습니다.")
            return
        # 아직 안됨
        # 아직 안됨
        # 아직 안됨
        # 제거할 도서
        book_to_remove = self._books[isbn]
        # 해당 도서가 대출 중일 경우
        if not book_to_remove.is_available:
            # 도서가 대출 중이면, 모든 회원의 대출 목록을 순회 후 해당 회원의 정보 출력
            for member_obj in self._members.values():
                # 해당 회원의 대출 목록에 도서가 있을 때
                # book_ot_remove 는 Book 객체
                # get_borrowd_books() 의 반환값은 스트링이 아니라 Book 객체
                if book_to_remove in member_obj.get_borrowed_books():
                    print(f"'{book_to_remove.title}' (ISBN: {isbn}) 도서는 현재 대출 중입니다.")
                    print(f"회원 '{member_obj.name}' (ID: {member_obj.member_id})님이 빌려갔습니다.")
                    print("도서가 반납된 후 다시 시도하십시오.")
                    return
        # isbn 값에 해당하는 도서가 있을 때
        del self._books[isbn]
        print(f"ISBN '{isbn}' 도서가 삭제되었습니다.")


    def search_book(self, query, type = "title"):
        """도서 검색(type 기본값: title)"""
        # 검색된 도서를 넣을 리스트
        results = []
        # 도서 제목으로 검색할 때
        if type == "title":
            # _books 딕셔너리의 값에는 Book 클래스 인스턴스가 있다.
            for book_obj in self._books.values():
                # 도서 제목이 일치할 때
                if query == book_obj.title:
                    results.append(book_obj)
        # 도서 저자로 검색할 때
        elif type == "author":
            for book_obj in self._books.values():
                # 도서 저자가 일치할 때
                if query == book_obj.author:
                    results.append(book_obj)
        # 출판연도로 검색할 때
        elif type == "year":
            for book_obj in self._books.values():
                # 도서 저자가 일치할 때
                if query == book_obj.year:
                    results.append(book_obj)
        # isbn으로 검색할 때
        elif type == "isbn":
            # self._books의 키가 isbn이므로 딕셔너리의 get 메서드 활용
            book_obj = self._books.get(query)
            # isbn의 값에 해당하는 도서가 존재할 때
            if book_obj:
                results.append(book_obj)
        # 검색 타입이 잘못된 경우
        else:
            print("유효하지 않은 검색 기준입니다.")
        return results

    def register_member(self, name):
        """회원 등록"""
        # 멤버 인스턴스 생성
        new_member = member_class.Member(name)
        # self._members 딕셔너리. 키: member_id, 값: Member 인스턴스
        # 이미 존재하는 회원일 경우
        if new_member.member_id in self._members:
            print(f"'{name}'님은 이미 등록된 회원입니다.")
            return
        self._members[new_member.member_id] = new_member
        print(f"'{name}'님이 성공적으로 등록되었습니다. 회원 ID: '{new_member.member_id}'")

        return new_member
    
    def remove_member(self, member_id):
        """회원 탈퇴"""
        # id 값에 해당하는 회원이 없을 경우
        if member_id not in self._members:
            print(f"회원 ID ({member_id})에 해당하는 회원이 없습니다.")
            return
        # id 값에 해당하는 회원이 대출 중인 도서가 있을 경우
        if self._members[member_id].get_borrowed_books():
            print(f"{self._members[member_id].name}, (회원 ID: {member_id})님은 아직 대출 중인 도서가 있습니다.")
            return
        # id 값에 해당하는 회원이 있을 경우
        print(f"'{self._members[member_id].name}' (회원 ID: {member_id})님이 탈퇴했습니다.")
        del self._members[member_id]

    def lend_book(self, member_id, isbn):
        """도서 대출(Library class)"""
        # Library 클래스에서 Member 클래스의 lend_book() 메서드 호출
        # 딕셔너리의 get() 메서드를 이용해 값(인스턴스) 가져오기
        member_obj = self._members.get(member_id)
        book_obj = self._books.get(isbn)

        # id에 해당하는 멤버가 없을 경우
        if not member_obj:
            print(f"회원 ID '{member_id}'를 찾을 수 없습니다.")
            return
        # isbn에 해당하는 도서가 없을 경우
        if not book_obj:
            print(f"ISBN '{isbn}'에 해당하는 도서를 찾을 수 없습니다.")
            return 
        # 해당 도서가 이미 대출중일 경우
        if not book_obj.is_available:
            print(f"'{book_obj.title}' 도서는 현재 대출 중입니다.")
            return
        
        # 대출에 성공했을 경우
        member_obj.lend_book(book_obj)
        book_obj.set_is_available = False

    def return_book(self, member_id, isbn):
        """도서 반납(Library class)"""
        # Library 클래스에서 Member 클래스의 return_book() 메서드 호출
        # 딕셔너리의 get() 메서드를 이용해 값(인스턴스) 가져오기
        member_obj = self._members.get(member_id)
        book_obj = self._books.get(isbn)

        # id에 해당하는 멤버가 없을 경우
        if not member_obj:
            print(f"회원 ID '{member_id}'를 찾을 수 없습니다.")
            return
        # isbn에 해당하는 도서가 없을 경우
        if not book_obj:
            print(f"ISBN '{isbn}'에 해당하는 도서를 찾을 수 없습니다.")
            return 
        # 해당 도서가 도서관에 있는 경우
        if book_obj.is_available:
            print(f"'{book_obj.title}' 도서는 이미 도서관에 있습니다.")
            return
        
        # 대출에 성공했을 경우
        member_obj.return_book(book_obj)
        book_obj.set_is_available = True

    def get_member_borrowed_books(self, member_id):
        """멤버의 도서 대출 목록 얻기"""
        member_obj = self._members.get(member_id)
        # id에 대항하는 멤버가 없을 경우
        if not member_obj:
            print(f"회원 ID '{member_id}'를 찾을 수 없습니다.")
            return
        # 해당 멤버의 도서 대출 리스트를 반환하는 함수 실행 후 반환
        return member_obj.get_borrowed_books()
    
    def display_all_books(self):
        """전체 도서 목록"""
        # 현재 등록된 도서가 없을 경우
        if not self._books:
            print("등록된 도서가 없습니다.")
        # 등록된 도서가 있을 경우
        else:
            for book_obj in self._books.values():
                status = "대출 가능" if book_obj.is_available else "대출 중"
                print(f"'{book_obj.title}' (저자: {book_obj.author}) (출판연도: {book_obj.year}) - {status}")
    
    def display_all_members(self):
        """전체 회원 목록"""
        # 현재 등록된 회원이 없을 경우
        if not self._members:
            print("등록된 회원이 없습니다.")
        # 등록된 회원이 있을 경우
        else:
            for member_obj in self._members.values():
                print(f"'{member_obj.name}' (회원 ID: {member_obj.member_id})")
                # 해당 회원이 현재 대출중인 도서가 없을 경우
                if not member_obj.get_borrowed_books():
                    print("현재 대출 중인 도서가 없습니다.")
                # 대출 중인 도서가 있을 경우
                else:
                    print(f"현재 대출 중인 도서 목록: {member_obj.get_borrowed_books()}")
                print()



# import book_class
# import member_class

# class Library:
#     """도서와 회원 컬렉션을 관리하고 대출/반납 기능을 제공하는 클래스입니다."""
#     def __init__(self):
#         self._books = {}
#         self._members = {}

#     def add_book(self, title, author, isbn, year):
#         if isbn in self._books:
#             print(f"❌ '{title}' (ISBN: {isbn}) 도서는 이미 등록되어 있습니다.")
#             return
#         new_book = book_class.Book(title, author, isbn, year)
#         self._books[isbn] = new_book
#         print(f"✅ '{title}' 도서가 성공적으로 추가되었습니다.")

#     def remove_book(self, isbn):
#         if isbn not in self._books:
#             print(f"❌ ISBN '{isbn}'에 해당하는 도서를 찾을 수 없습니다.")
#             return
#         del self._books[isbn]
#         print(f"✅ ISBN '{isbn}' 도서가 삭제되었습니다.")

#     def search_book(self, query, by='title'):
#         results = []
#         if by == 'title':
#             for book_obj in self._books.values():
#                 if query.lower() in book_obj.title.lower():
#                     results.append(book_obj)
#         elif by == 'author':
#             for book_obj in self._books.values():
#                 if query.lower() in book_obj.author.lower():
#                     results.append(book_obj)
#         elif by == 'isbn':
#             book_obj = self._books.get(query)
#             if book_obj:
#                 results.append(book_obj)
#         else:
#             print("❌ 유효하지 않은 검색 기준입니다.")
#         return results

#     def register_member(self, name):
#         new_member = member_class.Member(name)
#         if new_member.member_id in self._members:
#             print(f"❌ {name}님은 이미 등록된 회원입니다.")
#             return
#         self._members[new_member.member_id] = new_member
#         print(f"✅ {name}님이 성공적으로 등록되었습니다. 회원 ID: {new_member.member_id}")
#         return new_member

#     def lend_book(self, member_id, isbn):
#         member_obj = self._members.get(member_id)
#         book_obj = self._books.get(isbn)

#         if not member_obj:
#             print(f"❌ 회원 ID '{member_id}'를 찾을 수 없습니다.")
#             return
#         if not book_obj:
#             print(f"❌ ISBN '{isbn}'에 해당하는 도서를 찾을 수 없습니다.")
#             return

#         if not book_obj.is_available:
#             print(f"❌ '{book_obj.title}' 도서는 현재 대출 중입니다.")
#             return

#         member_obj.lend_book(book_obj)
#         book_obj.is_available = False

#     def return_book(self, member_id, isbn):
#         member_obj = self._members.get(member_id)
#         book_obj = self._books.get(isbn)
        
#         if not member_obj:
#             print(f"❌ 회원 ID '{member_id}'를 찾을 수 없습니다.")
#             return
#         if not book_obj:
#             print(f"❌ ISBN '{isbn}'에 해당하는 도서를 찾을 수 없습니다.")
#             return

#         if book_obj.is_available:
#             print(f"❌ '{book_obj.title}' 도서는 이미 도서관에 있습니다.")
#             return

#         member_obj.return_book(book_obj)
#         book_obj.is_available = True

#     def get_member_borrowed_books(self, member_id):
#         member_obj = self._members.get(member_id)
#         if not member_obj:
#             print(f"❌ 회원 ID '{member_id}'를 찾을 수 없습니다.")
#             return []
#         return member_obj.get_borrowed_books()

#     def display_all_books(self):
#         print("\n--- 도서관 전체 도서 목록 ---")
#         if not self._books:
#             print("등록된 도서가 없습니다.")
#         else:
#             for book_obj in self._books.values():
#                 status = "대출 가능" if book_obj.is_available else "대출 중"
#                 print(f"'{book_obj.title}' (저자: {book_obj.author}) - {status}")