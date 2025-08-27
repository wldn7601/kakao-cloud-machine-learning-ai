import queue
import threading
import time
import random

# 큐 생성
task_queue = queue.Queue()
result_queue = queue.Queue()

# 작업 생성 함수
def create_tasks(task_num):
    print("작업 생성 시작")

    # 작업 수만큼 작업 생성
    for i in range(task_num):
        task = f"작업-{i}"
        # 작업 큐에 추가
        task_queue.put(task)
        print(f"작업 추가: {task}")
        time.sleep(random.uniform(0.1, 0.3))
    
    # qsize() 함수의 부적확성으로 마지막에 None 추가
    for _ in range(3):
        task_queue.put(None)
    print("모든 작업 생성 완료")

# 작업 수행 함수
def worker(worker_id):
    # 현재 쓰레드
    print(f"{worker_id} 워커 시작")

    while True:
        # 작업 큐에서 작업 가져오기
        task = task_queue.get()
        # 작업이 전부 끝났었던 경우
        if task is None:
            print(f"{worker_id} 워커 종료")
            # 현재 작업이 끝났다는 코드. 전체 작업이 끝났다는 의미가 아님
            task_queue.task_done()
            break
        
        # 랜덤으로 시간을 쉬어서 간접적으로 프로세스 시간 얻기
        print(f"{worker_id} 원커 {task} 처리 중")
        processing_time = random.uniform(0.5, 1.5)
        time.sleep(processing_time)

        # 결과 형태 만들기
        result = f"{task} 소요 시간: {processing_time:.2f}"
        # 결과 큐에 (현재 쓰레드, 결과)를 튜플 형태로 추가
        result_queue.put((worker_id, result))

        # 현재 작업이 끝났다는 코드. 전체 작업이 끝났다는 의미가 아님
        task_queue.task_done()
        # 부정확해도 남은 작업 수 출력
        print(f"현재 남은 작업 수: {task_queue.qsize()}")

# 결과 수집기 함수
def result_collector():
    print("결과 수집 시작")
    # 수집한 결과를 넣을 리스트
    results = []

    while True:
        # 결과를 처리한 쓰레드, 결과를 언팩킹
        worker_id, result = result_queue.get()

        # 결과가 다 나왔었던 경우
        if worker_id is None and result is None:
            print(f"총 {len(results)}개 결과 수집")
            # 현재 결과가 끝났다는 코드. 전체 결과가 끝났다는 의미가 아님
            result_queue.task_done()
            break

        print(f"결과 수집: {worker_id} 워커 -> {result}")
        
        # 결과 리스트에 결과 추가
        results.append(result)
        # 현재 결과가 끝났다는 코드. 전체 결과가 끝났다는 의미가 아님
        result_queue.task_done()


# 쓰레드 생성
creator = threading.Thread(target=create_tasks , args=(100,))
# 컴프리헨션으로 3개 생성
workers = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
collector = threading.Thread(target=result_collector, args=())

# 쓰레드 동작 시작
creator.start()
for w in workers:
    w.start()
collector.start()

# 쓰레드가 끝날 때까지 기다림
creator.join()
for w in workers:
    w.join()
result_queue.put((None, None))
collector.join()

print("모든 작업 완료")
