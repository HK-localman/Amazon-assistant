# 定义函数a
def a(b):
    b()           # 执行传入的函数
    return "完成"  # 返回c（这里c是字符串）

# 定义函数b
def say_hello():
    print("Hello!")

# 调用a，传入b
result = a(say_hello)
# 输出: Hello!
print(result)  # 输出: 完成