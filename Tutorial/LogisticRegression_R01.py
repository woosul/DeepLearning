
import tensorflow as tf
import numpy as np

# 실행할 때마다 같은 값을 출력하기 위한 seed값 설정
seed = 0
np.random.seed(seed)
tf.set_random_seed(seed)

# x, y 데이터 값
x_data = np.array([[2, 3], [4, 3], [6, 4], [8, 6], [10, 7], [12, 8], [14, 9]])
y_data = np.array([0, 0, 0, 1, 1, 1, 1]).reshape(7, 1)

X = tf.placeholder(tf.float64, shape=[None, 2])
Y = tf.placeholder(tf.float64, shape=[None, 1])

# data = [[2, 0], [4, 0], [6, 0], [8, 1], [10, 1], [12, 1], [14, 1]]
# x_data = [x_row[0] for x_row in data]
# y_data = [y_row[1] for y_row in data]

a = tf.Variable(tf.random_uniform([2, 1], dtype=tf.float64))
b = tf.Variable(tf.random_uniform([1], dtype=tf.float64))

# sigmoid() 함수
# y = 1/(1+np.e**(a * x_data +b))
y = tf.sigmoid(tf.matmul(X, a) + b)

# 오차를 구하는 함수
# loss = -tf.reduce_mean(np.array(y_data) * tf.log(y) + (1 - np.array(y_data)) * tf.log(1-y))
loss = -tf.reduce_mean(Y * tf.log(y) + (1 - Y) * tf.log(1 - y))

# 경사하강법
learning_rate = 0.1
gradient_decent = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

predicted = tf.cast(y > 0.5, dtype=tf.float64)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float64))

# 학습
with tf.Session() as sess:
    sess.run(tf.global_variables_initializer())

    for i in range(3001):
        a_, b_, loss_, _ = sess.run([a, b, loss, gradient_decent], feed_dict={X: x_data, Y: y_data})

        if (i + 1) % 300 == 0:
            print("step = %d, a1 = %.4f, a2 = %.4f, b = %.4f, loss = %.4f" % (i + 1, a_[0], a_[1], b_, loss_))
