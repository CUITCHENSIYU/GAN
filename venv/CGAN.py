from __future__ import print_function,division

from keras.datasets import mnist
from keras.layers import Input, Dense, Reshape, Flatten, Dropout, multiply
from keras.layers import BatchNormalization, Activation, Embedding, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import Adam

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from keras.models import load_model
from keras.models import model_from_json


#输入样本的形状
img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
num_classes = 10
latent_dim = 100

#构建生成器模型
def build_generator():
        model = Sequential()

        model.add(Dense(256, input_dim=latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(img_shape), activation='tanh'))
        model.add(Reshape(img_shape))

        model.summary()

        noise = Input(shape=(latent_dim,))
        label = Input(shape=(1,), dtype='int32')
        label_embedding = Flatten()(Embedding(num_classes, latent_dim)(label))#Embedding：将标签转为one-hot编码，即10*100矩阵

        model_input = multiply([noise, label_embedding])
        img = model(model_input)

        return Model([noise, label], img)

#构建鉴别器模型
def build_discriminator():
        model = Sequential()

        model.add(Dense(512, input_dim=np.prod(img_shape)))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dropout(0.4))
        model.add(Dense(1, activation='sigmoid'))
        model.summary()

        img = Input(shape=img_shape)
        label = Input(shape=(1,), dtype='int32')

        label_embedding = Flatten()(Embedding(num_classes, np.prod(img_shape))(label))#np.prod计算所有元素的乘积，
        flat_img = Flatten()(img)

        model_input = multiply([flat_img, label_embedding])

        validity = model(model_input)

        return Model([img, label], validity)

#选择优化器
optimizer = Adam(0.0002, 0.5)

#构建并编译判别器
discriminator = build_discriminator()
discriminator.compile(loss=['binary_crossentropy'],
                                   optimizer=optimizer,
                                   metrics=['accuracy'])
#构建生成器
generator = build_generator()
#生成器的输入为噪音z和分类标签label,输出是标签指定分类的假图像
noise = Input(shape=(latent_dim,))
label = Input(shape=(1,))
img = generator([noise, label])

#对于组合模型，我们只训练生成器
discriminator.trainable = False

#鉴别器将生成的图像作为输入并确定该图像的有效性和标签
valid = discriminator([img, label])

#组合模型，训练生成器去欺骗鉴别器
combined = Model([noise, label], valid)
combined.compile(loss=['binary_crossentropy'],
                              optimizer=optimizer)##################################################

#训练
def train(epochs, batch_size=128, sample_interval=50):
        #加载数据
        (X_train, y_train), (_, _) = mnist.load_data("MNIST_data")

        #配置输入
        X_train = (X_train.astype(np.float32) - 127.5) / 127.5
        X_train = np.expand_dims(X_train, axis=3)
        y_train = y_train.reshape(-1, 1)

        ##判别器判别后的两种结果：valid和fake，真为1，假为0
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        # 定义可视化操作
        sess = tf.Session()
        writer = tf.summary.FileWriter('/temp/data/logs', sess.graph)
        #D_loss = tf.placeholder(tf.float32)
        G_loss = tf.placeholder(tf.float32)
        #tf.summary.scalar("D_loss", D_loss)
        tf.summary.scalar("G_loss", G_loss)
        merged = tf.summary.merge_all()
        i = 0
        ##############################
        #开始训练
        ##############################
        for epoch in range(epochs):
            # ---------------------
            #  训练鉴别器
            # ---------------------


            #随机获取半个batch的数据集，且每个样本的索引标号为0-50000
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs, labels = X_train[idx], y_train[idx]#因为我们的条件是label，所以需要输入的样本和样本标号相一致，即又标号指定输入哪个样本，imgs:128*32*32*3，labels:128*1

            #获得噪音，输入给生成器(128*100)
            noise = np.random.normal(0, 1, (batch_size, 100))

            #生成新图像
            gen_imgs = generator.predict([noise, labels])

            if i%5==0:

                #开始训练鉴别器
                d_loss_real = discriminator.train_on_batch([imgs, labels], valid)########################################
                d_loss_fake = discriminator.train_on_batch([gen_imgs, labels], fake)
                d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)
            i += 1

            # ---------------------
            #  训练生成器
            # ---------------------
            #标签上的条件
            sampled_labels = np.random.randint(0, 10, batch_size).reshape(-1, 1)
            #训练生成器
            g_loss = combined.train_on_batch([noise, sampled_labels], valid)#################################################

            #打印训练过程
            print("%d [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100 * d_loss[1], g_loss))

            #每过一个sample_interval保存一张图片
            if epoch % sample_interval == 0:
                sample_images(epoch)

            # 将训练结果保存到可视化数据
            summary = sess.run(merged, feed_dict={G_loss: g_loss})
            writer.add_summary(summary, epoch)


#保存图片
def sample_images( epoch):
        r, c = 2, 5
        noise = np.random.normal(0, 1, (r * c, 100))
        sampled_labels = np.arange(0, 10).reshape(-1, 1)

        gen_imgs = generator.predict([noise, sampled_labels])

        # Rescale images 0 - 1
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
                axs[i, j].set_title("Digit: %d" % sampled_labels[cnt])
                axs[i, j].axis('off')
                cnt += 1
        fig.savefig("E:/DeepLearn/myproject/generator_%d.png" % epoch)
        plt.close()

train(epochs=5000,batch_size=128,sample_interval=100)