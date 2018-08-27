
import tensorflow as tf

# x1, x2, y 데이터 값
data = [[2, 0, 81], [4, 4, 93], [6, 2, 91], [8, 3, 97]]
x1 = [x_row1[0] for x_row1 in data]
x2 = [x_row2[1] for x_row2 in data]
y_data = [y_row[2] for y_row in data]

# 기울기 a 와 y절편 b값을 랜덤에 의해 임의로 지정
# 단, 기울기의 범위는 0~10사이, y절편은 0~100사이에서 추출
a1 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
a2 = tf.Variable(tf.random_uniform([1], 0, 10, dtype=tf.float64, seed=0))
b = tf.Variable(tf.random_uniform([1], 0, 100, dtype=tf.float64, seed=0))

# y에 대한 일차방정식 a1x1 + a2x2 + b의 식
y = a1 * x1 + a2 * x2 + b

# Tensorflow RMSE 함수
rmse = tf.sqrt(tf.reduce_mean(tf.square(y - y_data)))

# 학습율 값
learning_rate = 0.1

# RMSE값을 최소로 하는 값 찾기
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(rmse)

# Tensorflow를 이용한 학습
with tf.Session() as sess:

    # 변수 초기화
    sess.run(tf.global_variables_initializer())

    # 2001번 실행 (0번째를 포함)
    for step in range(2001):
        sess.run(gradient_decent)

        # 100번마다 결과 출력
        if step % 100 == 0:
            print("Epoch: %.f, RMSE = %.04f, Slope a1 = %.4f, Slope a2 = %.4f, y Start b = %.4f" %
                  (step, sess.run(rmse), sess.run(a1), sess.run(a2), sess.run(b)))


