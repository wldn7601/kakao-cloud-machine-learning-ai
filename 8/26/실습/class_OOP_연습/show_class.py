# import product_class
# import user_class
# import order_class
import data_class

class Show:
    """상품, 유저를 보여주는 클래스"""
    def __init__(self, data: data_class.Data) -> None:
        """Data 인스턴스"""
        self.products = data.products
        self.users = data.users
    
    # 특정 유저 검색
    def search_user(self, user_id):
        """유저 id. 해당 유저의 정보를 출력하는 메서드"""
        # 해당 id인 유저가 없을 경우
        if user_id not in self.users:
            print(f"ID: {user_id} 유저가 없습니다.")
            return
        # 유저 인스턴스
        user = self.users[user_id]

        # 유저 정보 출력

        print(f"### ID: {user_id} ###")
        print("현재 장바구니: ", end="")
        shopping_cart_is_empty = True
        # 해당 유저의 장바구니의 상품 정보, 개수 출력
        for value in user.shopping_cart.values():
            # value: {'product': Product 인스턴스, 'count': 숫자}
            print(f"{value['product']}, {value['count']}개")
            shopping_cart_is_empty = False
        # 장바구니에 아무것도 없을 경우
        if shopping_cart_is_empty:
            print("비어 있습니다.")

        print("현재 구매 목록: ", end="")
        order_list_is_empty = True
        # 해당 유저의 구매 목록의 상품 정보, 개수 출력
        for value in user.order_list.values():
            # value: {'product': Product 인스턴스, 'count': 숫자}
            print(f"{value['product']}, {value['count']}개")
            order_list_is_empty = False
        # 장바구니에 아무것도 없을 경우
        if order_list_is_empty:
            print("비어 있습니다.")

    # 전체 유저 출력
    def display_all_users(self):
        """전체 유저 정보를 출력하는 메서드"""
        # 현재 등록된 유저가 없는 경우
        if not self.users:
            print("현재 등록된 유저가 없습니다.")
            return
        # 현재 등록된 유저가 있는 경우
        for user_id in self.users.keys():
            # 현재 id인 유저를 검색해서 사용
            # search_user() 메서드 재사용
            self.search_user(user_id)
            
    # 특정 상품 검색
    def search_product(self, product_name):
        """상품 이름. 해당 상품의 정보를 출력하는 메서드"""
        # 해당 상품이 상품 목록에 없는 경우
        if product_name not in self.products:
            print(f"{product_name} 상품은 상품 목록에 없습니다.")
            return
        # 상품 인스턴스
        product = self.products[product_name]
        # 상품 정보 출력
        print(product)

    # 전체 상품 출력
    def display_all_products(self):
        """전체 상품 정보를 출력하는 메서드"""
        # 현재 등록된 상품이 없는 경우
        if not self.products:
            print("현재 등록된 상품이 없습니다.")
            return
        # 현재 등록된 상품이 있는 경우
        for product_name in self.products.keys():
            # 현재 상품 이름을 검색해서 사용
            # search_product() 메서드 재사용
            self.search_product(product_name)