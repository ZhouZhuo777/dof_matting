# 这是一个示例 Python 脚本。

# 按 ⌃R 执行或将其替换为您的代码。
# 按 双击 ⇧ 在所有地方搜索类、文件、工具窗口、操作和设置。
from shapely.geometry import  Point, LineString, Polygon
import shapely.affinity

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

# print(len(combinations(n, arr)))

rect = Polygon([(0,0),(0,1),(1.99,1),(1,0)])
rect1 = Polygon([(2,0),(2,1),(3,1),(3,0)])
print(rect.area)
print(rect1.area)
print(rect1.intersects(rect))

circle=Point(0,0).buffer(1) #type(circle)=polygon
print(circle.area)
print(rect.intersects(circle))

circle1 = Point(2,0).buffer(0.9)
ellipse = shapely.affinity.scale(circle1,1.2,1.5) #type(ellipse)=polygon
ellipse1 = shapely.affinity.rotate(ellipse,45) #type(ellipse)=polygon
print(ellipse1.area)
print(ellipse)
print(ellipse1)
print(circle.intersects(ellipse1))

circle11=Point(2072,887).buffer(200)
print(circle11.area)

circle1 = Point(2036.83,584.82).buffer(1)
ellipse_11 = shapely.affinity.scale(circle1,175,299.61) #type(ellipse)=polygon
ellipse1_11 = shapely.affinity.rotate(ellipse_11,89.58)

print(ellipse1_11.area)
print(circle11.intersects(ellipse1_11))
print(ellipse1_11.intersects(circle11))

rect_test = Polygon([(360,0),(360,401),(1279,401),(1279,0)])
# rect_test = Polygon([(2,0),(2,1),(3,1),(3,0)])
print(rect_test.area)
print(rect_test)
# 2072 887 200
# 2036.83 584.82   175 299.61    89.58