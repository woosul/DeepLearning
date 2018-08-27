
import numpy as np

# 기울기 a, y절편 b
ab = [3, 76]

# x,y의 데이터값
data = [[2, 81], [4, 93], [6, 91], [8, 97]]
x = [i[0] for i in data]
y = [i[1] for i in data]


# y=ax+b에 x,a,b를 대입하여 예측결과값 y를 리턴하는 함수
def predict(x):
    return ab[0]*x + ab[1]


# RMSE(Root Mean Squared Error) 함수
def rmse(p, a):
    return np.sqrt(((p - a) ** 2).mean())


# RMSE 함수를 y값에 대입하는 최종값을 구하는 함수
def rmse_val(predict_result, y):
    return rmse(np.array(predict_result), np.array(y))


# 예측값이 들어갈 빈 리스트
predict_result = []


# 모든 x값을 한번씩 대입 예측값 리스트를 완성
for i in range(len(x)):
    predict_result.append(predict(x[i]))
    print("공부한시간 = %.f, 실제점수 = %.f, 예측점수 = %.f" % (x[i], y[i], predict(x[i])))


# 최종 RMSE 출력
print("rmse 최종값 : " + str(rmse_val(predict_result, y)))
