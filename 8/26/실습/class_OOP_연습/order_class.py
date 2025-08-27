# import product_class
# import user_class
import data_class

class Order:
    """상품을 주문하는 클래스"""
    def __init__(self, data: data_class.Data) -> None:
        """Data 인스턴스"""
        # Data 인스턴스의 상품 딕셔너리
        self.products = data.products
        # Data 인스턴스의 유저 딕셔너리
        self.users = data.users
    
    # 장바구니에 담기
    def add_to_shopping_cart(self, product_name, user_id):
        """상품 이름, 유저 id. 유저의 장바구니에 상품 추가"""
        # 해당 상품이 상품 목록에 없는 경우
        if product_name not in self.products:
            print(f"{product_name} 상품이 상품 목록에 없습니다.")
            return 
        # 해당 유저가 없는 경우
        if user_id not in self.users:
            print(f"ID: {user_id} 유저가 없습니다.")
            return
        # 해당 유저의 장바구니에 상품 추가
        # 만약 해당 상품을 이미 장바구니에 추가한 경우
        if product_name in self.users[user_id].shopping_cart:
            # 해당 상품의 카운트 + 1
            self.users[user_id].shopping_cart[product_name]['count'] += 1
            print(f"ID: {user_id}, {product_name} 상품이 장바구니에 추가되었습니다.")
            return
        # 해당 상품이 장바구니에 처음 추가된 경우
        new_dict = {
            'product': self.products[product_name],
            'count': 1
        }
        # shopping_cart는 이중 딕셔너리
        self.users[user_id].shopping_cart[product_name] = new_dict
        print(f"ID: {user_id}, {product_name} 상품이 장바구니에 추가되었습니다.")

        
    # 장바구니에서 상품 제거
    def remove_shopping_cart(self, product_name, user_id):
        """상품 이름, 유저 id. 해당 유저의 해당 상품 장바구니에서 제거"""
        # 해당 id인 유저가 없을 경우
        if user_id not in self.users:
            print(f"유저 ID: {user_id} 유저가 없습니다.")
            return
        # 해당 상품 이름이 상품 목록에 없는 경우
        if product_name not in self.products:
            print(f"{product_name} 상품이 상품 목록에 없습니다.")
            return
        # 해당 유저가 해당 상품을 장바구니에 추가했는지 검사해야 한다.
        is_contained = False
        for value in self.users[user_id].shopping_cart.values():
            # value: {'product': Product 인스턴스, 'count': 숫자}
            # 해당 유저가 해당 상품을 장바구니에 추가한 경우
            if product_name == value['product'].name:
                is_contained = True
                break
        # 해당 유저가 해당 상품을 장바구니에 추가하지 않았을 경우
        if not is_contained:
            print(f"ID: {user_id} 유저는 {product_name}을 장바구니에 추가하지 않았습니다.")
            return
        # 해당 유저가 해당 상품을 장바구니에 추가한 경우. 장바구니에서 제거
        del self.users[user_id].shopping_cart[product_name]
        print(f"ID: {user_id}. {product_name} 상품을 장바구니에서 제거했습니다.")
    
    # 장바구니 전체 가격 계산
    def calc_shopping_cart(self, user_id):
        """유저 id. 해당 유저의 장바구니 가격 계산"""
        # 해당 id인 유저가 없을 경우
        if user_id not in self.users:
            print(f"ID: {user_id} 유저가 없습니다.")
            return
        # 해당 id인 유저가 있을 경우
        total_price = 0
        # 해당 유저의 장바구니
        user_shopping_cart = self.users[user_id].shopping_cart
        for value in user_shopping_cart.values():
            # 현재 value 상태: {'product' : Product 인스턴스, 'count' : 숫자}
            total_price += value['product'].price * value['count']
        print(f"### ID: {user_id} ###")
        print(f"총 가격: {total_price}")

    # 현재 장바구니 전체 구매
    def buy_products(self, user_id):
        """우저 id. 해당 유저의 장바구니 전체 구매"""
        # 해당 id인 유저가 없을 경우
        if user_id not in self.users:
            print(f"ID: {user_id} 유저가 없습니다.")
            return
        # 해당 유저
        user = self.users[user_id]
        # 현재 장바구니 전체 가격 계산
        self.calc_shopping_cart(user_id)

        # 동일한 상품이 이미 있으면 count 숫자 값 증가
        # 없으면 새로 추가
        # 먼저 장바구니에서 상품 이름을 가져와 구매 목록의 상품 이름과 비교
        for product_name in user.shopping_cart:
            # 구매 목록에 있으면
            if product_name in user.order_list:
                # 구매 개수를 더한다.
                user.order_list[product_name]['count'] += user.shopping_cart[product_name]['count']
            # 구매 목록에 없으면
            else:
                # 새로 추가
                user.order_list[product_name] = user.shopping_cart[product_name]

        # 해당 유저의 shoppring_cart 비우기
        user.shopping_cart.clear()
        
  