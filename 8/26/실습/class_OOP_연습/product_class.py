# 상품 정보: 상품이름, 가격, 카테고리, 제조사
class Product:
    """상품 정보를 담는 클래스"""
    def __init__(self, name, price, categoty, manufacturer) -> None:
        """상품 이름(유일), 가격, 카테고리, 제조사"""
        self.name = name
        self.price = price
        self.categoty = categoty
        self.manufacturer = manufacturer

    def __repr__(self) -> str:
        return f"Product(상품 이름: {self.name}, 가격: {self.price}, 카테고리: {self.categoty}, 제조사: {self.manufacturer})"