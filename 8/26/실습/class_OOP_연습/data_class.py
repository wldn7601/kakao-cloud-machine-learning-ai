# 데이터 클래스
import product_class
import user_class

class Data:
    """상품, 유저 데이터를 담는 클래스"""
    def __init__(self) -> None:
        # 가입한 유저. id가 키고, 값이 User 인스턴스
        self.users = {}
        # 입고한 상품들. 상품 이름이 키고, 값이 Product 인스턴스
        self.products = {}
    
    # 상품 추가
    def add_product(self, name, price, category, manufacturer):
        """상품 이름, 가격, 카테고리, 제조사. 상품 등록 메서드"""
        # 이미 추가된 상품일 경우
        if name in self.products:
            print(f"상품: {name}은(는) 이미 추가된 상품입니다.")
            return
        # 해당 상품이 없을 경우. 상품 추가
        product = product_class.Product(name, price, category, manufacturer)
        self.products[name] = product
        print(f"상품: {name}이(가) 추가되었습니다.")

    # 상품 제거
    # 유저가 장바구니에 넣은 상품도 같이 제거 해야됨 
    def remove_product(self, name):
        """상품 이름. 상품 제거 메서드"""
        # 해당 이름의 상품이 상품 목록에 없을 경우
        if name not in self.products:
            print(f"{name} 상품이 상품 목록에 없습니다.")
            return
        # 유저가 이미 장바구니에 해당 상품을 추가했을 때 상품을 목록에서 제거하면 
        # 해당 유저의 장바구니에도 적용해야 한다.
        for user_id in self.users:
            user = self.users[user_id]
            # 해당 상품이 장바구니에 있다면
            if name in user.shopping_cart:
                # 삭제
                del user.shopping_cart[name]
                print(f"ID: {user_id} 유저의 장바구니에서 {name} 상품이 제거되었습니다.")
        # 해당 이름의 상품이 있을 경우. 상품 제거
        del self.products[name]
        print(f"{name} 상품이 제거되었습니다.")
        
    # 유저 등록
    def register_user(self, id, password):
        """유저 id, password. 유저 등록 메서드"""
        # 이미 등록된 id일 경우
        if id in self.users:
            print(f"{id}은(는) 이미 사용 중인 ID입니다.")
            return
        # 해당 id가 없을 경우. 유저 추가
        user = user_class.User(id, password)
        self.users[id] = user
        print(f"ID: {id} 유저가 등록되었습니다.")

    # 유저 탈퇴
    def remove_user(self, id):
        """유저 id. 유저 탈퇴 메서드"""
        # 해당 id의 유저가 없을 경우
        if id not in self.users:
            print(f"ID: {id} 유저가 없습니다.")
            return
        # 해당 id의 유저가 있을 경우
        del self.users[id]
        print(f"ID: {id} 유저가 제거되었습니다.")