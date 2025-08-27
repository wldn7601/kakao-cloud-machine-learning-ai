import data_class
import show_class
import order_class

def main_menu():
    """사용자에게 메뉴를 보여주고 입력에 따라 함수를 호출합니다."""
    data = data_class.Data()
    show = show_class.Show(data)
    order = order_class.Order(data)

    print("--- 시스템 시작 ---")
    print("-" * 20)

    while True:
        print("\n[메뉴] 1.상품추가 2.상품삭제 3.상품검색 4.전체상품 5.유저등록 6.유저삭제 7.유저검색 8.전체유저")
        print("[메뉴] 9.장바구니에 담기 10.장바구니 상품 제거 11.장바구니 가격 계산 12.장바구니 전체 구매 13.종료\n")
        choice = input("원하는 메뉴 번호를 입력하세요: ")

        # 상품 추가
        if choice == '1':
            print("\n--- 상품 추가 ---")
            # 상품 정보 입력
            product_name = input("상품 이름: ")
            product_price = input("가격: ")
            # 가격이 정수로 변환이 안될 때
            if not product_price.isdecimal():
                print(f"{product_price}는 유요하지 않는 가격입니다.")
                continue
            # 정수로 변환 가능
            product_price = int(product_price)
            product_category = input("카테고리: ")
            product_manufacturer = input("제조사: ")
            data.add_product(product_name, product_price, product_category, product_manufacturer)

        
        # 상품 삭제
        elif choice == '2':
            print("\n--- 상품 삭제 ---")
            product_name = input("상품 이름: ")
            data.remove_product(product_name)
        
        # 상품 검색
        elif choice == '3':
            print("\n--- 상품 검색 ---")
            product_name = input("검색할 상품 이름: ")
            show.search_product(product_name)
            
        
        # 전체 상품 출력
        elif choice == '4':
            print("\n--- 전체 상품 ---")
            show.display_all_products()
            

        # 유저 등록
        elif choice == '5':
            print("\n--- 유저 등록 ---")
            user_id = input("ID: ")
            user_password = input("PassWord: ")
            data.register_user(user_id, user_password)

        # 유저 삭제
        elif choice == '6':
            print("\n--- 유저 삭제 ---")
            user_id = input("ID: ")
            data.remove_user(user_id)

        # 유저 검색
        elif choice == '7':
            print("\n--- 유저 검색 ---")
            user_id = input("ID: ")
            show.search_user(user_id)

        # 전체 유저 출력
        elif choice == '8':
            print("\n--- 전체 유저 ---")
            show.display_all_users()

        # 장바구니 담기
        elif choice == '9':
            print("\n--- 장바구니 담기 ---")
            product_name = input("상품 이름: ")
            user_id = input("ID: ")
            order.add_to_shopping_cart(product_name, user_id)

        # 장바구니 상품 제거
        elif choice == '10':
            print("\n--- 장바구니 상품 제거 ---")
            product_name = input("상품 이름: ")
            user_id = input("ID: ")
            order.remove_shopping_cart(product_name, user_id)

        # 장바구니 전체 가격 계산
        elif choice == '11':
            print("\n--- 장바구니 전체 가격 계산 ---")
            user_id = input("ID: ")
            order.calc_shopping_cart(user_id)

        # 장바구니 전체 상품 구매
        elif choice == '12':
            print("\n--- 장바구니 전체 상품 구매 ---")
            user_id = input("ID: ")
            order.buy_products(user_id)

        # 종료
        elif choice == '13':
            print("\n--- 종료 ---\n")
            break
        
        # 잘못된 메뉴 번호
        else:
            print("잘못된 메뉴 번호입니다. 다시 입력해 주세요.")
            
if __name__ == "__main__":
    main_menu()