import pickle as p
from keras.layers import Input,Dense,Reshape,Flatten,Dropout
from keras.layers import BatchNormalization,Activation,ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D,Conv2D
from keras.models import Sequential,Model
from keras.optimizers import Adam
from keras import backend as k
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from keras.datasets import mnist

# 输入图片尺寸为（28,28,1）
# 输入数据为100维：100*z
img_rows=28
img_cols=28
channels=1
img_shape=(img_rows,img_cols,channels)
latent_dim=100

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
    model.add(Dense(1,activation='sigmoid'))

    model.summary()

    img=Input(shape=img_shape)#这里作为鉴别器入口
    validity=model(img)#这里作为出口,输出判定结果

    return Model(img,validity)




#初始化优化器，搭建并编译鉴别器
optimizer=Adam(0.0002,0.5)

discriminator=build_discriminator()
discriminator.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

#编译生成器
generator=build_generator()
z=Input(shape=(100,))
img=generator(z)
#注：在训练生成器的时候，鉴别器设置为不可训练，并且需要将判别器与生成器相连
discriminator.trainable=False
valid=discriminator(img)
combined=Model(z,valid)#拼接模型的输入与输出分别为：z,valid
# 这里也可以自定义损失函数
#见：https://github.com/bojone/gan/blob/master/keras/dcgan_celeba.py
combined.compile(loss='binary_crossentropy',optimizer=optimizer)

def train(batch_size=128):
    (X_train,_),(_,_)=mnist.load_data("MNIST_data")
    X_train=X_train/127.5-1.#将数值限制在一个范围内（-1,1）
    X_train=np.expand_dims(X_train,axis=3)#扩展维度在3轴上?????????????????????????????????

    valid=np.ones((batch_size,1))#????????????????????????????
    fake=np.zeros((batch_size,1))
    # 定义可视化操作
    sess = tf.Session()
    writer = tf.summary.FileWriter('/temp/data/logs', sess.graph)
    D_loss = tf.placeholder(tf.float32)
    G_loss = tf.placeholder(tf.float32)
    tf.summary.scalar("D_loss", D_loss)
    tf.summary.scalar("G_loss", G_loss)
    merged = tf.summary.merge_all()

    for epoch in range(1000):
        idx=np.random.randint(0,X_train.shape[0],batch_size)#返回一个随机整形数，范围从0-X_train.shape[0]，尺寸为batch_size
        imgs=X_train[idx]
        noise=np.random.normal(0,1,(batch_size,latent_dim))#获取一个形状为(batch_size,latent_dim)的正态分布随机数
        gen_imgs=generator.predict(noise)

        d_loss_real=discriminator.train_on_batch(imgs,valid)#模型的输入与目标输出为imgs和valid  注意：这里的valid也可能是Y_train
        d_loss_fake=discriminator.train_on_batch(gen_imgs,fake)
        d_loss=0.5*np.add(d_loss_real,d_loss_fake)

        g_loss=combined.train_on_batch(noise,valid)
        print("batch %d d_loss  : %f  acc：%.2f%%" % (epoch,d_loss[0],100*d_loss[1]))
        print("g_loss : %f" % (g_loss))

        if epoch % 100 == 0:

            sample_images(epoch)
        # 将训练结果保存到可视化数据

        summary = sess.run(merged, feed_dict={ G_loss: g_loss,D_loss:d_loss})
        writer.add_summary(summary, epoch)


def sample_images(epoch):
    r, c = 2, 5
    noise = np.random.normal(0, 1, (r * c, 100))

    gen_imgs = generator.predict(noise)

    # Rescale images 0 - 1
    gen_imgs = 0.5 * gen_imgs + 0.5

    fig, axs = plt.subplots(r, c)
    cnt = 0
    for i in range(r):
        for j in range(c):
            axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
            axs[i, j].axis('off')
            cnt += 1
    fig.savefig("E:/DeepLearn/myproject/generator_%d.png" % epoch)
    plt.close()


train()




