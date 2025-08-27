# 유저 클래스
import product_class

class User:
    """유저 정보를 담는 클래스"""
    def __init__(self, id, password) -> None:
        """id(유일), password"""
        self.id = id
        # shopping_cart: 딕셔너리로 저장, 키: 상품 이름, 값: {product: Product 인스턴스, count: 상품 개수}
        # 이중 딕셔너리
        self.shopping_cart = {}
        # order_list: 딕셔너리로 저장, 키: 상품 이름, 값: {product: Product 인스턴스, count: 상품 개수}
        # 이중 딕셔너리
        self.order_list = {}
        self._password = password

    def __repr__(self) -> str:
        return f"유저 ID: {self.id}, 현재 장바구니: {self._shopping_cart}, 구매한 상품: {self._order_list}"
    