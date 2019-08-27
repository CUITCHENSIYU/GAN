from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply, concatenate
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D, Lambda
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam
from keras.utils import to_categorical
import keras.backend as K

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np

#定义图片形状
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
#定义类别数量
num_classes = 10
#定义隐藏维度
latent_dim = 72

###################
#定义损失函数
#损失函数1：交叉熵损失函数：用于真假判定
#损失函数2：自定义：用于互信息
####################
def mutual_info_loss(c, c_given_x):
    """The mutual information metric we aim to minimize"""
    eps = 1e-8
    conditional_entropy = K.mean(- K.sum(K.log(c_given_x + eps) * c, axis=1))
    entropy = K.mean(- K.sum(K.log(c + eps) * c, axis=1))

    return conditional_entropy + entropy
losses = ['binary_crossentropy', mutual_info_loss]

#定义优化器
optimizer = Adam(0.0002, 0.5)

#构建判别器模型和识别模型，因为我们要求判别器具有还原隐含编码和判别真伪的能力
def build_disk_and_q_net():
    img = Input(shape=img_shape)

    # 判别器和识别网络共享网络层及参数
    model = Sequential()
    model.add(Conv2D(64, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(512, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())

    img_embedding = model(img)

    # 判别器
    validity = Dense(1, activation='sigmoid')(img_embedding)

    # 识别
    q_net = Dense(128, activation='relu')(img_embedding)
    label = Dense(num_classes, activation='softmax')(q_net)

    # 返回判别网络和识别网络
    return Model(img, validity), Model(img, label)

#构建生成器
def build_generator():
    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=3, padding="same"))
    model.add(Activation("relu"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(channels, kernel_size=3, padding='same'))
    model.add(Activation("tanh"))

    gen_input = Input(shape=(latent_dim,))
    img = model(gen_input)

    model.summary()

    return Model(gen_input, img)

#定义判别器和辅助网络
discriminator, auxilliary = build_disk_and_q_net()

#编译判别器
discriminator.compile(loss=['binary_crossentropy'],
            optimizer=optimizer,
            metrics=['accuracy'])

#编译辅助网络
auxilliary.compile(loss=[mutual_info_loss],
            optimizer=optimizer,
            metrics=['accuracy'])

#定义生成器
generator = build_generator()

#生成器使用噪音和标签作为输入，并且生成标签指定数字
gen_input = Input(shape=(latent_dim,))#输入
img = generator(gen_input)#输出

#在组合模型中我们将只训练生成器
discriminator.trainable = False

#判别器以生成图片作为输入，并判断真假
valid = discriminator(img)

#由辅助网络生成图像标签
target_label = auxilliary(img)

#组合模型（判别器和生成器,Q）
combined = Model(gen_input, [valid, target_label])
combined.compile(loss=losses,optimizer=optimizer)

#定义样本输入,输入到生成器的噪音和隐含编码，并将类别用矩阵表示
def sample_generator_input(batch_size):
    # Generator inputs
    sampled_noise = np.random.normal(0, 1, (batch_size, 62))
    sampled_labels = np.random.randint(0, num_classes, batch_size).reshape(-1, 1)
    sampled_labels = to_categorical(sampled_labels, num_classes=num_classes)

    return sampled_noise, sampled_labels

#定义训练样本
def train(epochs, batch_size=128, sample_interval=50):
    #获取数据集
    (X_train, y_train), (_, _) = mnist.load_data("MNIST_data")
    #将数据集调整为{-1,1}
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    y_train = y_train.reshape(-1, 1)

    #定义真假变量（real:1,fake:0）
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # 定义可视化操作
    sess = tf.Session()
    writer = tf.summary.FileWriter('/temp/data/logs', sess.graph)
    # D_loss = tf.placeholder(tf.float32)
    G_loss = tf.placeholder(tf.float32)
    # tf.summary.scalar("D_loss", D_loss)
    # G_loss=tf.reduce_mean(G_loss,name='G_loss')
    tf.summary.histogram("G_loss", G_loss)
    merged = tf.summary.merge_all()

    #开始迭代训练
    for epoch in range(epochs):
        # ---------------------
        #  训练判别器
        # ---------------------
        #获取训练集
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs = X_train[idx]
        # 噪音z和隐含编码，并将噪音和隐含编码
        sampled_noise, sampled_labels = sample_generator_input(batch_size)
        gen_input = np.concatenate((sampled_noise, sampled_labels), axis=1)

        #生成图片
        gen_imgs = generator.predict(gen_input)

        # 在真实图片和生成图片进行训练
        d_loss_real = discriminator.train_on_batch(imgs, valid)
        d_loss_fake = discriminator.train_on_batch(gen_imgs, fake)
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  训练生成器和Q辅助网络
        # ---------------------
        #定义生成器损失函数
        g_loss = combined.train_on_batch(gen_input, [valid, sampled_labels])

        # 打印过程
        print("%d [D loss: %.2f, acc.: %.2f%%] [Q loss: %.2f] [G loss: %.2f]" % (
        epoch, d_loss[0], 100 * d_loss[1], g_loss[1], g_loss[2]))

        # 当达到一定迭代次数，保存一张图片
        if epoch % sample_interval == 0:
            sample_images(epoch)
        # 将训练结果保存到可视化数据
        summary = sess.run(merged, feed_dict={G_loss: g_loss})
        writer.add_summary(summary, epoch)

#保存图片
def sample_images( epoch):
    r, c = 10, 10

    fig, axs = plt.subplots(r, c)
    for i in range(c):
        sampled_noise, _ = sample_generator_input(c)
        label = to_categorical(np.full(fill_value=i, shape=(r, 1)), num_classes=num_classes)
        gen_input = np.concatenate((sampled_noise, label), axis=1)
        gen_imgs = generator.predict(gen_input)
        gen_imgs = 0.5 * gen_imgs + 0.5
        for j in range(r):
            axs[j, i].imshow(gen_imgs[j, :, :, 0], cmap='gray')
            axs[j, i].axis('off')
    fig.savefig("E:/DeepLearn/myproject/generate_%d.png" % epoch)
    plt.close()

def save_model():

    def save(model, model_name):
        model_path = "E:/DeepLearn/myproject/InfoGANmodel/%s.json" % model_name
        weights_path = "E:/DeepLearn/myproject/InfoGANmodel/%s_weights.hdf5" % model_name
        options = {"file_arch": model_path,
                       "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])

    save(generator, "generator_model")
    save(discriminator, "discriminator_model")

train(epochs=5000, batch_size=128, sample_interval=50)



