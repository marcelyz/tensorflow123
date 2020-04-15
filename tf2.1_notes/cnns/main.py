import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from resnet18 import ResNet18


np.set_printoptions(threshold=np.inf)

# data
fashion = tf.keras.datasets.fashion_mnist
(x_train, y_train), (x_test, y_test) = fashion.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("x_train.shape: ", x_train.shape)
x_train = x_train.reshape(x_train.shape[0], 28, 28, 1)  # 给数据增加一个单通道维度，使数据和网络结构匹配
x_test = x_test.reshape(x_test.shape[0], 28, 28, 1)
print("x_train.shape: ", x_train.shape)


# model = Baseline()

# LeNet: 卷积网络开篇之作，共享卷积核，减少网络参数
# model = LeNet5()

# AlexNet: 使用relu激活函数，提升训练速度；使用Dropout，缓解过拟合
# model = AlexNet8()

# VGGNet: 小尺寸卷积核减少参数，网络结构规整，适合并行加速
# model = VGG16()

# InceptionNet: 一层内使用不同尺寸卷积核，提升感知力；使用批标准化，缓解梯度消失
# model = Inception10(num_blocks=2, num_classes=10)

# ResNet: 层间残差跳连，引入前方信息，缓解模型退化，使神经网络层数加深成为可能
model = ResNet18([2, 2, 2, 2])

model.compile(optimizer='adam',
              loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
              metrics=['sparse_categorical_accuracy'])

history = model.fit(x_train, y_train, batch_size=32, epochs=5, validation_data=(x_test, y_test))
model.summary()


# save weights
file = open('./weights.txt', 'w')
for v in model.trainable_variables:
    file.write(str(v.name) + '\n')
    file.write(str(v.shape) + '\n')
    file.write(str(v.numpy()) + '\n')
file.close()


# show
acc = history.history['sparse_categorical_accuracy']
val_acc = history.history['val_sparse_categorical_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

plt.subplot(1, 2, 1)
plt.plot(acc, label='Training Accuracy')
plt.plot(val_acc, label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss, label='Training Loss')
plt.plot(val_loss, label='Validation Loss')
plt.title('Training and Validation Loss')
plt.legend()
plt.show()
