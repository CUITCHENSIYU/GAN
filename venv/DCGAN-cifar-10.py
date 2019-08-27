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
from keras.preprocessing import image
import os
import pickle


latent_dim=32
height=32
width=32
channels=3
img_shape=(height,width,channels)

def build_generateor():
    model=Sequential()
    noise=Input(shape=(latent_dim,))
    model.add(Dense(128*16*16,input_dim=latent_dim))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Reshape((16,16,128)))

    model.add(Conv2D(256,kernel_size=5,padding='same'))
    model.add(LeakyReLU(alpha=0.2))
    model.add(UpSampling2D())
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256,kernel_size=5,padding='same'))
    model.add(LeakyReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(256,kernel_size=5,padding='same'))
    model.add(LeakyReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(channels,kernel_size=7,activation='tanh',padding='same'))

    model.summary()
    img=model(noise)
    return Model(noise,img)
def build_discriminator():
    model=Sequential()

    img=Input(shape=img_shape)
    model.add(Conv2D(128,kernel_size=3,input_shape=img_shape))
    model.add(LeakyReLU())
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128,kernel_size=4,strides=2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=4, strides=2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Conv2D(128, kernel_size=4, strides=2))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Flatten())
    model.add(Dropout(0.4))

    model.add(Dense(1,activation='sigmoid'))

    model.summary()

    vaild=model(img)
    return Model(img,vaild)

#加载数据集
def load_CIFAR10_batch(filename):
    with open(filename, 'rb') as f:
        data = pickle.load(f, encoding='latin1')
        X = data['data']
        y = data['labels']
        X = X.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1).astype(np.float32)
        y = np.array(y)
        return X, y
def load_CIFAR10(root):
    xs, ys = [], []
    for n in range(1, 6):
        filename = os.path.join(root, 'data_batch_{}'.format(n))
        X, y = load_CIFAR10_batch(filename)
        xs.append(X)
        ys.append(y)
    Xtr = np.concatenate(xs)
    Ytr = np.concatenate(ys)
    Xte, Yte = load_CIFAR10_batch(os.path.join(root, 'test_batch'))
    return (Xtr, Ytr),( Xte, Yte)

optimizer=Adam(0.0002,0.5)

discriminator=build_discriminator()
discriminator=build_discriminator()
discriminator.compile(loss="binary_crossentropy",optimizer=optimizer,metrics=['accuracy'])

generator=build_generateor()
z=Input(shape=(latent_dim,))
gen_img=generator(z)
discriminator.trainable=False
valid=discriminator(gen_img)
combined=Model(z,valid)
combined.compile(loss='binary_crossentropy',optimizer=optimizer)

def train(epochs,batch_size):
    (x_train, y_train), (_, _) = load_CIFAR10('E:/DeepLearn/myproject/venv/cifar-10-batches-py')

    #从数据集中选择frog类（class 6）
    x_train=x_train[y_train.flatten()==6]

    #标准化数据
    x_train=x_train.reshape((x_train.shape[0],)+(height,width,channels)).astype('float32')/255.

    save_dir='E:/DeepLearn/myproject/gen_img'
    #定义真假标签
    valid = np.ones((batch_size, 1))
    fake = np.zeros((batch_size, 1))
    start=0

    # 定义可视化操作
    sess = tf.Session()
    writer = tf.summary.FileWriter('/temp/data/logs', sess.graph)
    D_loss = tf.placeholder(tf.float32)
    G_loss = tf.placeholder(tf.float32)
    tf.summary.scalar("D_loss", D_loss)
    tf.summary.scalar("G_loss", G_loss)
    merged = tf.summary.merge_all()

    for step in range(epochs):
        noise = np.random.normal(0, 1, (batch_size, latent_dim))
        gen_imgs=generator.predict(noise)

        stop=start+batch_size
        real_imgs=x_train[start:stop]
        #训练鉴别器
        d_loss_real=discriminator.train_on_batch(real_imgs,valid)
        d_loss_fake=discriminator.train_on_batch(gen_imgs,fake)
        d_loss=0.5*np.add(d_loss_real,d_loss_fake)

        #训练生成器
        g_loss=combined.train_on_batch(noise,valid)
        #下一批次
        start+=batch_size
        if start>len(x_train)-batch_size:
            start=0
        print('step: %d  g_loss: %s  d_loss: %s   ' % (step, g_loss, d_loss))
        if step%500==0:
            #保存生成的图像
            img=image.array_to_img(gen_imgs[0]*255.,scale=False)
            img.save(os.path.join(save_dir,'gen_frog'+str(step)+'.png'))

        summary = sess.run(merged, feed_dict={G_loss: g_loss, D_loss: d_loss})
        writer.add_summary(summary, step)

train(1000,32)






