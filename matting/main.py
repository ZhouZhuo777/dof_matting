# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。

def combinations(n, arr):
    result = []
    helper(n, arr, [], result)
    return result

def helper(n, arr, temp, result):
    if n == 0:
        result.append(temp)
        return
    for i in range(len(arr)):
        helper(n-1, arr[i+1:], temp+[arr[i]], result)


arr = ['1','2','3','4','5','6','7','8','9']
n=5

print(len(combinations(n, arr)))