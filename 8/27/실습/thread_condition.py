import threading
import time

data = None
# 컨디션 생성
condition = threading.Condition()

# 데이터를 기다리는 함수
def wait_for_data():
    print("대기 스레드: 데이터 대기 중...")

    # 락 획득
    with condition:
        # 알림을 받을 때까지 기다림
        # 받으면 락을 얻고 계속 실행
        condition.wait()
        print(f"{data} 데이터를 수신했습니다.")

# 데이터를 준비하는 함수
def prepare_data():
    global data
    print(f"준비 스레드: 데이터 준비 중...")
    time.sleep(2)

    # 락 획득
    with condition:
        data = "준비된 데이터"
        print("데이터가 준비되었습니다.")
        # 데이터가 준비되었다고 알림을 준다.
        condition.notify()

# 쓰레드 생성
t1 = threading.Thread(target=wait_for_data)
t2 = threading.Thread(target=prepare_data)

# 쓰레드 시작
t1.start()
t2.start()

# 쓰레드가 끝날 때까지 기다림
t1.join()
t2.join()
