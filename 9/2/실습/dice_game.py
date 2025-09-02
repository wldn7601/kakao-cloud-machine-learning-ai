
def solution():
    count = 0
    results = []
    results_set = set()
    while count < 4:
        n = input("주사위 숫자(1 ~ 6): ")

        if not n.isdecimal() or n < '1' or n > '6':
            print("유효한 숫자를 입력하세요")
            continue
        
        n = int(n)
        count += 1
        results.append(n)
        results_set.add(n)
   
    results_set_size = len(results_set)

    # (4)
    if results_set_size == 1:
        print("(4)")
        return 1111 * results_set.pop()
    # (3, 1), (2, 2)
    elif results_set_size == 2:
        p = results_set.pop()
        q = results_set.pop()
        # (2, 2)
        if results.count(results[0]) == 2:
            print("(2, 2)")
            return (p + q) * abs(p - q)
        # (3, 1)
        else:
            print("(3, 1)")
            if results.count(q) == 3:
                p,q = q,p
            return (10 * p + q) ** 2
    # (2, 1, 1)
    elif results_set_size == 3:
        print("(2, 1, 1)")
        r = 1
        for num in results:
            if results.count(num) == 1:
                r *= num
        return r
    # (1, 1, 1, 1)
    else:
        print("(1, 1, 1, 1)")
        return min(results_set)
        

print(solution())


