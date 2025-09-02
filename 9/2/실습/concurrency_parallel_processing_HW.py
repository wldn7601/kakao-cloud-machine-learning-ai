import asyncio
import aiohttp
import time
import concurrent.futures
import requests

urls = [
    "https://jsonplaceholder.typicode.com/posts/1",
    "https://jsonplaceholder.typicode.com/posts/2",
    "https://jsonplaceholder.typicode.com/posts/3",
    "https://jsonplaceholder.typicode.com/posts/4",
    "https://jsonplaceholder.typicode.com/posts/5"
]

# 순차 처리
async def sequential_fetch(session, url):
    try:
        start_time = time.time()
        print(f"GET 요청 시작: {url}")
        async with session.get(
            url,
            timeout=10,
            params={"_": f"{time.time()}"},  # 캐시 회피용 쿼리 파라미터
            headers={"Cache-Control": "no-cache"}
        ) as response:
            text = await response.text()
            elapsed = time.time() - start_time
            print(f"응답 완료: {url} status={response.status} {response.reason} bytes={len(text)} elapsed={elapsed:.2f}초")
            return url, len(text), elapsed
    except Exception as e:
        print(f"{url} 오류 발생: {e}")
        return url, 0, 0.0

async def sequential_main(urls):
    start_time = time.time()
    results = [] # asyncio, aiothhp 와 바뀐 부분

    async with aiohttp.ClientSession() as session:
        for url in urls: # asyncio, aiothhp 와 바뀐 부분
            result = await sequential_fetch(session, url) # asyncio, aiothhp 와 바뀐 부분
            results.append(result) # asyncio, aiothhp 와 바뀐 부분

    end_time = time.time()
    duration = end_time - start_time
    print(f"총 소요시간(sequential): {duration:.2f}초")
    return results, duration
# 순차 처리

# threadPoolExecutor
def thread_fetch(url: str):
    try:
        start_time = time.time()
        print(f"GET 요청 시작: {url}")
        response = requests.get(url, timeout=10, headers={"Cache-Control": "no-cache"}, params={"_": f"{time.time()}"})
        text = response.text
        elapsed = time.time() - start_time
        print(f"응답 완료: {url} status={response.status_code} {response.reason} bytes={len(text)} elapsed={elapsed:.2f}초")
        return url, response.status_code, len(text), elapsed
    except Exception as e:
        print(f"오류: {url} error={e}")
        return url, 0, 0, 0.0

def thread_main(urls):
    start_time = time.time()
    print(f"ThreadPoolExecutor 시작 (workers={5})")

    results = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as executor:
        future_to_url = {executor.submit(thread_fetch, url): url for url in urls}
        for future in concurrent.futures.as_completed(future_to_url):
            result = future.result()
            results.append(result)

    end_time = time.time()
    duration = end_time - start_time
    print(f"총 소요시간(threadPoolExecutor): {duration:.2f}초")
    total_bytes = sum(size for _, _, size, _ in results)
    print(f"총 바이트 수: {total_bytes}")
    return results, duration
# threadPoolExecutor


# asyncio, aiohttp
async def aiohttp_fetch(session, url):
    try:
        start_time = time.time()
        print(f"GET 요청 시작: {url}")
        async with session.get(
            url,
            timeout=10,
            params={"_": f"{time.time()}"},  # 캐시 회피용 쿼리 파라미터
            headers={"Cache-Control": "no-cache"}
        ) as response:
            text = await response.text()
            elapsed = time.time() - start_time
            print(f"응답 완료: {url} status={response.status} {response.reason} bytes={len(text)} elapsed={elapsed:.2f}초")
            return url, len(text), elapsed
    except Exception as e:
        print(f"{url} 오류 발생: {e}")
        return url, 0, 0.0

async def aiohttp_main(urls):
    start_time = time.time()
    
    async with aiohttp.ClientSession() as session:
        tasks = [aiohttp_fetch(session, url) for url in urls]
        results = await asyncio.gather(*tasks)

    end_time = time.time()
    duration = end_time - start_time
    print(f"총 소요시간(async, aiohttp): {duration:.2f}초")
    return results, duration
# asyncio, aiohttp



if __name__ == "__main__":
    print("=== 세 가지 방식 비교 시작 ===")

    # 순차 처리
    print("\n[1] 순차 처리 (순차 처리)\n")
    seq_results, seq_duration = asyncio.run(sequential_main(urls))
    seq_total_bytes = sum(size for _, size, _ in seq_results)
    print(f"순차 처리 총 바이트: {seq_total_bytes}")
    time.sleep(1)

    # ThreadPoolExecutor
    print("\n[2] ThreadPoolExecutor (requests)\n")
    th_results, th_duration = thread_main(urls)
    th_total_bytes = sum(size for _, _, size, _ in th_results)
    print(f"쓰레드풀 총 바이트: {th_total_bytes}")
    time.sleep(1)


    # asyncio + aiohttp (병렬)
    print("\n[3] asyncio + aiohttp (병렬)\n")
    aio_results, aio_duration = asyncio.run(aiohttp_main(urls))
    aio_total_bytes = sum(size for _, size, _ in aio_results)
    print(f"aiohttp 총 바이트: {aio_total_bytes}")

    print("\n--- 요약 (초) ---")
    print(f"sequential: {seq_duration:.2f}")
    print(f"threadPool: {th_duration:.2f}")
    print(f"aiohttp   : {aio_duration:.2f}")
    