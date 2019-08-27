from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np


#定义一些形状
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
num_classes = 10
latent_dim = 100

#定义优化器和所需损失函数
optimizer = Adam(0.0002, 0.5)
losses = ['binary_crossentropy', 'sparse_categorical_crossentropy']

#构建生成器模型
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

    model.summary()

    noise = Input(shape=(latent_dim,))#输入噪音
    label = Input(shape=(1,), dtype='int32')#输入标签
    label_embedding = Flatten()(Embedding(num_classes, 100)(label))#将标签大小变为和噪音一样

    model_input = multiply([noise, label_embedding])#向量乘法，作为输入
    img = model(model_input)

    return Model([noise, label], img)
#构建判别器模型
def build_discriminator():
    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0, 1), (0, 1))))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))

    model.add(Flatten())
    model.summary()

    img = Input(shape=img_shape)

    # 提取特征
    features = model(img)

    # 确定图片的真假和类别
    validity = Dense(1, activation="sigmoid")(features)
    label = Dense(num_classes, activation="softmax")(features)

    return Model(img, [validity, label])

#构建并且编译判别器
discriminator = build_discriminator()
discriminator.compile(loss=losses,optimizer=optimizer,metrics=['accuracy'])

#构建生成器
generator = build_generator()

#生成器使用噪音和标签作为输入，并生成标签指定数字
noise = Input(shape=(latent_dim,))
label = Input(shape=(1,))#########################################
img = generator([noise, label])

# 对于组合模型，我们只训练生成器
discriminator.trainable = False

#鉴别器将生成的图像作为输入并确定该图像的有效性和标签
valid, target_label = discriminator(img)

# 组合模型，训练生成器去欺骗鉴别器
combined = Model([noise, label], [valid,target_label])
combined.compile(loss=losses,optimizer=optimizer)#为什么这里的Loss是一个包含两个损失函数的数组，因为我们要计算两个损失，分别是判别器判断真假和判断类别


################
#训练函数
################
def train(epochs, batch_size=128, sample_interval=50):
    #加载数据
    (X_train, y_train), (_, _) = mnist.load_data("MNIST_data")

    #配置输入
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    y_train = y_train.reshape(-1, 1)

    #判别器判别后的两种结果：valid和fake，真为1，假为0
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))

    # 定义可视化操作
    sess = tf.Session()
    writer = tf.summary.FileWriter('/temp/data/logs', sess.graph)
    # D_loss = tf.placeholder(tf.float32)
    G_loss = tf.placeholder(tf.float32)
    # tf.summary.scalar("D_loss", D_loss)
    #G_loss=tf.reduce_mean(G_loss,name='G_loss')
    tf.summary.histogram("G_loss", G_loss)
    merged = tf.summary.merge_all()

    #开始迭代训练epochs次
    for epoch in range(epochs):
        # ---------------------
        #  训练鉴别器
        # ---------------------

        # 获取半个batch的数据集
        idx = np.random.randint(0, X_train.shape[0], batch_size)
        imgs, labels = X_train[idx], y_train[idx]

        # 获得噪音，输入给生成器
        noise = np.random.normal(0, 1, (batch_size, 100))

        #生成器尝试创建sample_lable所指定的图片,sample_label即条件
        sampled_labels = np.random.randint(0, 10, (batch_size, 1))###############################
        #生成图片
        gen_imgs = generator.predict([noise, sampled_labels])

        #图片标签：0-9
        img_labels = y_train[idx]

        #开始训练鉴别器：定义损失函数
        d_loss_real = discriminator.train_on_batch(imgs, [valid, img_labels])
        d_loss_fake = discriminator.train_on_batch(gen_imgs, [fake, sampled_labels])
        d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

        # ---------------------
        #  训练生成器
        # ---------------------

        #训练生成器：定义损失函数：两个损失函数
        g_loss = combined.train_on_batch([noise, sampled_labels], [valid, sampled_labels])##########################

        #打印训练过程
        print("%d [D loss: %f, acc.: %.2f%%, op_acc: %.2f%%] [G loss: %f]" % (
        epoch, d_loss[0], 100 * d_loss[3], 100 * d_loss[4], g_loss[0]))

        # 当达到sanple_interval次 => 保存生成图片
        if epoch % sample_interval == 0:
            save_model()
            sample_images(epoch)

            # 将训练结果保存到可视化数据
            summary = sess.run(merged, feed_dict={G_loss: g_loss})
            writer.add_summary(summary, epoch)
#保存图片
def sample_images(epoch):
    r, c = 10, 10
    noise = np.random.normal(0, 1, (r * c, 100))
    sampled_labels = np.array([num for _ in range(r) for num in range(c)])
    gen_imgs = generator.predict([noise, sampled_labels])
    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("E:/DeepLearn/myproject/generate_%d.png" % epoch)
    plt.close()

#################
#保存训练模型
##################
def save_model():

    def save(model, model_name):
        model_path = "E:/DeepLearn/myproject/model/%s.json" % model_name
        weights_path = "E:/DeepLearn/myproject/model/%s_weights.hdf5" % model_name
        options = {"file_arch": model_path,"file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])

    save(generator, "generator_model")
    save(discriminator, "discriminator_model")

#开始训练
train(epochs=140, batch_size=32, sample_interval=200)