import numpy as np
import tensorflow as tf

# tensor: represent an array of 0 to n order
c = np.arange(24).reshape(2, 4, 3)
print(c)

# tf tensor
a = tf.constant([1, 5], dtype=tf.int64)
print(a)
print(a.dtype)
print(a.shape)

# numpy format to tf tensor
a = np.arange(0, 5)
b = tf.convert_to_tensor(a, dtype=tf.int64)
print(a)
print(b)

# tensor of different values
a = tf.zeros([2, 3])
b = tf.ones(4)
c = tf.fill([2, 2], 9)
print(a)
print(b)
print(c)

# tensor of different distribution random
d = tf.random.normal([2, 2], mean=0.5, stddev=1)
print(d)
e = tf.random.truncated_normal([2, 2], mean=0.5, stddev=1)
print(e)
f = tf.random.uniform([2, 2], minval=0, maxval=1)
print(f)

# common functions
x1 = tf.constant([1., 2., 3.], dtype=tf.float64)
print(x1)
x2 = tf.cast(x1, tf.int32)
print(x2)

x = tf.constant([[1, 2, 3], [2, 2, 3]])
print(x)
print(tf.reduce_mean(x))
print(tf.reduce_sum(x, axis=1))

w = tf.Variable(tf.random.normal([2, 2], mean=0, stddev=1))

a = tf.ones([1, 3])
b = tf.fill([1, 3], 3.)
print(a)
print(b)
print(tf.add(a, b))
print(tf.subtract(a, b))
print(tf.multiply(a, b))
print(tf.divide(b, a))

a = tf.fill([1, 2], 3.)
print(a)
print(tf.pow(a, 3))
print(tf.square(a))
print(tf.sqrt(a))

a = tf.ones([3, 2])
b = tf.fill([2, 3], 3.)
print(tf.matmul(a, b))

features = tf.constant([12, 23, 10, 17])
labels = tf.constant([0, 1, 1, 0])
dataset = tf.data.Dataset.from_tensor_slices((features, labels))
print(dataset)
for element in dataset:
    print(element)

with tf.GradientTape() as tape:
    w = tf.Variable(tf.constant(3.0))
    loss = tf.pow(w, 2)
grad = tape.gradient(loss, w)
print(grad)

classes = 3
labels = tf.constant([1, 0, 2])
output = tf.one_hot(labels, depth=classes)
print(output)

y = tf.constant([1.01, 2.01, -0.66])
y_pro = tf.nn.softmax(y)
print("After softmax, y_pro is: ", y_pro)

w = tf.Variable(4)
w.assign_sub(1)
print(w)

test = np.array([[1, 2, 3], [2, 3, 4], [5, 4, 3], [8, 7, 2]])
print(test)
print(tf.argmax(test, axis=0))
print(tf.argmax(test, axis=1))

# forward
x1 = tf.constant([[5.8, 4.0, 1.2, 0.2]])
w1 = tf.constant([[-0.8, -0.34, -1.4],
                  [0.6, 1.3, 0.25],
                  [0.5, 1.45, 0.9],
                  [0.65, 0.7, -1.2]])
b1 = tf.constant([2.52, -3.1, 5.62])
y = tf.matmul(x1, w1) + b1
print("x1.shape: ", x1.shape)
print("w1.shape: ", w1.shape)
print("b1.shape: ", b1.shape)
print("y.shape: ", y.shape)
print("y: ", y)

y_dim = tf.squeeze(y)
y_pro = tf.nn.softmax(y_dim)
print("y_dim: ", y_dim)
print("y_pro: ", y_pro)

# back propagation
w = tf.Variable(tf.constant(5, dtype=tf.float32))
lr = 0.2
epoch = 40
for epoch in range(epoch):
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
    grads = tape.gradient(loss, w)
    w.assign_sub(lr * grads)
    print("After %s epoch, w is %f, loss is %s " % (epoch, w.numpy(), loss))