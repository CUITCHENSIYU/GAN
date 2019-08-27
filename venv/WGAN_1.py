#WGAN和DCGAN的区别在于以下几点，所以有以下修改点：
#1：WGAN的代价函数并不存在log，
#2：WGAN的目标在于测试生成数据分布于真实数据分布之间的距离，而非原始GAN的是与否的二分类问题，故除去判别器最后一层的sigmoid函数
#3：因为要使用Lipschitz，所以需要加上权重剪裁使网络参数保持在一定范围内。
#4：将Adam等梯度下降方法改为使用RMSProp方法。
from keras.datasets import mnist
from keras.layers import Input,Dense,Reshape,Flatten,Dropout
from keras.layers import BatchNormalization,Activation,ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D,Conv2D
from keras.models import Sequential,Model
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.optimizers import RMSprop

import keras.backend as k


# 输入图片尺寸为（28,28,1）
# 输入数据为100维：100*z
img_rows=28
img_cols=28
channels=1
img_shape=(img_rows,img_cols,channels)
latent_dim=100
n_discriminator=5
clip_value=0.01

#定义kasserstein损失
def wasserstein_loss(y_true,y_pred):
    return k.mean(y_true*y_pred)

def build_generator():
    model=Sequential()

    model.add(Dense(128*7*7,activation="relu",input_dim=latent_dim))
    model.add(Reshape((7,7,128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64,kernel_size=3,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(channels,kernel_size=3,padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise=Input(shape=(latent_dim,))#这里作为生成器的入口
    img=model(noise)#这里作为出口

    return Model(noise,img)

def build_discriminator():
    model=Sequential()

    model.add(Conv2D(32,kernel_size=3,strides=2,input_shape=img_shape,padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64,kernel_size=3,strides=2,padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128,kernel_size=3,strides=2,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(256,kernel_size=3,strides=1,padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))###################修改点一

    model.summary()

    img=Input(shape=img_shape)#这里作为鉴别器入口
    validity=model(img)#这里作为出口,输出判定结果

    return Model(img,validity)

#初始化优化器，搭建并编译鉴别器
optimizer=RMSprop(lr=0.00005)#######################修改点二

discriminator=build_discriminator()
discriminator.compile(loss=wasserstein_loss,optimizer=optimizer,metrics=['accuracy'])#################修改点四

generator=build_generator()
z=Input(shape=(100,))
img=generator(z)
#注：在训练生成器的时候，鉴别器设置为不可训练，并且需要将判别器与生成器相连
discriminator.trainable=False
valid=discriminator(img)
combined=Model(z,valid)
###########################################修改点三
combined.compile(loss=wasserstein_loss,metrics=['accuracy'],optimizer=optimizer)

def train(epochs,batch_size):
    (X_train,y_train),(_,y_test)=mnist.load_data("MNIST_data")
    X_train=X_train/127.5-1.
    X_train=np.expand_dims(X_train,axis=3)

    valid=np.ones((batch_size,1))
    fake=np.zeros((batch_size,1))

    # 定义可视化操作
    sess = tf.Session()
    writer = tf.summary.FileWriter('/temp/data/logs', sess.graph)
    #D_loss = tf.placeholder(tf.float32)
    G_loss = tf.placeholder(tf.float32)
    #tf.summary.scalar("D_loss", D_loss)
    tf.summary.histogram('G_loss', G_loss)
    merged = tf.summary.merge_all()

    for epoch in range(epochs):
        for _ in range(n_discriminator):

            idx=np.random.randint(0,X_train.shape[0],batch_size)
            imgs=X_train[idx]
            noise=np.random.normal(0,1,(batch_size,latent_dim))
            gen_imgs=generator.predict(noise)

            d_loss_real=discriminator.train_on_batch(imgs,valid)
            d_loss_fake=discriminator.train_on_batch(gen_imgs,fake)
            d_loss=0.5*np.add(d_loss_real,d_loss_fake)
            for l in discriminator.layers:
                weights=l.get_weights()
                weights=[np.clip(w,-clip_value,clip_value) for w in weights]
                l.set_weights(weights)

        g_loss=combined.train_on_batch(noise,valid)#损失函数接收的是以noise作为输入产生的gen_img和valid,比较二者的差距
        print("batch %d d_loss :" % (epoch))
        print(d_loss)
        print("batch %d g_loss : %s" % (epoch, g_loss))

        g_loss=np.array(g_loss)

        # 将训练结果保存到可视化数据
        summary = sess.run(merged, feed_dict={G_loss:g_loss})
        writer.add_summary(summary, epoch)
train(100,128)




