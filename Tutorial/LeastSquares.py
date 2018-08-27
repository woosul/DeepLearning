# -*- coding: utf-8 -*-

import numpy as np

# x, y 값
x = [2, 4, 6, 8]
y = [81, 93, 91, 97]

# x, y의 평균값
mx = np.mean(x)
my = np.mean(y)
print("x의 평균값 : ", mx)
print("y의 평균값 : ", my)

# 기울기 공식의 분모
divisor = sum([(mx - i)**2 for i in x])


# 기울기 공식의 분자
def top(x, mx, y, my):
    d = 0
    for i in range(len(x)):
        d += (x[i] - mx) * (y[i] - my)
    return d


dividend = top(x, mx, y, my)
print("기울기 분모 : ", divisor)
print("기울기 분자 : ", dividend)

# 기울기 a와 y절편 b
a = dividend / divisor
b = my - (mx * a)

# 결과값 확인
print("예측공식 : y = ", a, "x +", b)
print("기울기 (a) : ", a)
print("y절편 (b) : ", b)

