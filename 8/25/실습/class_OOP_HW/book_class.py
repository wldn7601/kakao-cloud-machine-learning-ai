class Book:
    """도서 정보(제목, 저자, ISBN, 출판연도)를 관리하는 클래스입니다."""
    def __init__(self, title, author, isbn, year):
        # 공개 정보
        self.title = title
        self.author = author
        self.year = year
        # 비공개 정보
        self._isbn = isbn
        self._is_available = True
    
    # 인스턴스.isbn 형태로 정보 가져오기 가능
    @property
    def isbn(self):
        return self._isbn
    @property
    def is_available(self):
        return self._is_available
    # is_available의 값을 바꾸는 메서드
    @is_available.setter
    def set_is_available(self, status):
        self._is_available = status

    # print(인스턴스)를 했을 때 인스턴스의 주소가 나오는 대신
    # 반환한 값이 나온다.
    def __repr__(self) -> str:
        return f"Book(제목: '{self.title}', 저자: '{self.author}', ISBN: '{self.isbn}', 출판연도: '{self.year}')"

# class Book:
#     """도서 정보(제목, 저자, ISBN, 출판연도)를 관리하는 클래스입니다."""
#     def __init__(self, title, author, isbn, year):
#         self.title = title
#         self.author = author
#         self.year = year
#         self._isbn = isbn
#         self._is_available = True

#     @property
#     def isbn(self):
#         return self._isbn

#     @property
#     def is_available(self):
#         return self._is_available

#     @is_available.setter
#     def is_available(self, status):
#         self._is_available = status
        
#     def __repr__(self):
#         return f"Book(제목: '{self.title}', 저자: '{self.author}', ISBN: '{self.isbn}')"