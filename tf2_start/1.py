import tensorflow as tf

print(tf.__version__)

X = tf.constant(range(12))
print(X.shape)
print(X)

trs_x = tf.reshape(X, (3, 4))
print(trs_x)

Y = tf.constant([[2, 1, 4, 3], [1, 2, 3, 4], [4, 3, 2, 1]])

print(Y)

random_x = tf.random.normal(shape=[3, 4], mean=0, stddev=1)
# numberic

a = trs_x + Y
b = trs_x * Y
c = trs_x / Y

Y = tf.cast(Y, tf.float32)
y_e = tf.exp(Y)

# matrix
Y = tf.cast(Y, tf.int32)
tf.matmul(trs_x, tf.transpose(Y))

# concat
xy = tf.concat([trs_x, Y], axis=0)
print(xy)
xy2 = tf.concat([trs_x, Y], axis=1)
print(xy2)

print(trs_x == Y)

print(tf.reduce_sum(trs_x))
print(tf.norm(tf.cast(X, tf.float32)))
