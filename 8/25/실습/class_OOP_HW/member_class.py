import book_class
import uuid

class Member:
    """도서관 회원 정보와 대출 목록을 관리하는 클래스입니다."""
    def __init__(self, name) -> None:
        # 공개 정보
        self.name = name
        # 비공개 정보
        self._member_id = str(uuid.uuid4())
        self._borrowed_books = []
    @property
    def member_id(self):
        return self._member_id

    def lend_book(self, book_obj: book_class.Book):
        """도서 대출(Member class)"""
        self._borrowed_books.append(book_obj)
        print(f"'{self.name}'님이 '{book_obj.title}'을(를) 대출했습니다.")

    def return_book(self, book_obj: book_class.Book):
        """도서 반납(Member class)"""
        # 도서 대출 목록에 없을 경우
        if book_obj not in self._borrowed_books:
            print(f"{self.name}님은 '{book_obj.title}'을(를) 대출한 기록이 없습니다.")
        # 도서 대출 목록에 있을 경우
        else:
            self._borrowed_books.remove(book_obj)
            print(f"{self.name}님이 '{book_obj.title}'을(를) 반납했습니다.")
    
    # _borrowed_books 게터
    def get_borrowed_books(self):
        return self._borrowed_books
    
    def __repr__(self) -> str:
        return f"Member(이름: '{self.name}', ID: '{self.member_id}')"


# import uuid
# import book_class

# class Member:
#     """도서관 회원 정보와 대출 목록을 관리하는 클래스입니다."""
#     def __init__(self, name):
#         self.name = name
#         self._member_id = str(uuid.uuid4())
#         self._borrowed_books = []

#     @property
#     def member_id(self):
#         return self._member_id

#     def lend_book(self, book_obj: book_class.Book):
#         self._borrowed_books.append(book_obj)
#         print(f"✅ {self.name}님이 '{book_obj.title}'을(를) 대출했습니다.")

#     def return_book(self, book_obj: book_class.Book):
#         if book_obj in self._borrowed_books:
#             self._borrowed_books.remove(book_obj)
#             print(f"✅ {self.name}님이 '{book_obj.title}'을(를) 반납했습니다.")
#         else:
#             print(f"❌ {self.name}님은 '{book_obj.title}'을(를) 대출한 기록이 없습니다.")

#     def get_borrowed_books(self):
#         return self._borrowed_books

#     def __repr__(self):
#         return f"Member(이름: '{self.name}', ID: '{self.member_id}')"