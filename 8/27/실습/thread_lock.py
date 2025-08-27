import threading
import time

counter = 0
# 락 생성
counter_Lock = threading.Lock()

# counter에 1을 더하는 함수
def add_one(times):
    global counter
    for _ in range(times):
        # 자동으로 락 생성 및 해제
        with counter_Lock:
            current = counter
            time.sleep(0.001)
            counter = current + 1

# 쓰레드 생성
t1 = threading.Thread(target=add_one, args=(1000,))
t2 = threading.Thread(target=add_one, args=(1000,))

# 쓰레드 시작
t1.start()
t2.start()

# 쓰레드가 끝날 때까지 기다림
t1.join()
t2.join()

# 데이터 공유와 동기화가 일어났는지 확인
print(counter)