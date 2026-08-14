def climb_stairs(n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    return climb_stairs(n-1) + climb_stairs(n-2)

print(climb_stairs(10))  # 输出 89

def fib(n):
    if n == 1 or n == 2:          # 只判断 n==1
        return 1
    return fib(n-1) + fib(n-2)

print (fib(11))

def climb_stairs(n):
    if n == 1:
        return 1
    if n == 2:
        return 2
    
    prev2 = 1  # f(1)
    prev1 = 2  # f(2)
    
    for i in range(3, n+1):
        current = prev1 + prev2  # f(i) = f(i-1) + f(i-2)
        prev2 = prev1
        prev1 = current
    
    return current

print(climb_stairs(10))  # 输出 89

climb_stairs = lambda n: n if n <= 2 else climb_stairs(n-1) + climb_stairs(n-2)
print(climb_stairs(10))