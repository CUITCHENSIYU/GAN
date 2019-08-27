from __future__ import print_function, division

from keras.datasets import mnist
from keras.layers.merge import _Merge
from keras.layers import Input, Dense, Reshape, Flatten, Dropout
from keras.layers import BatchNormalization, Activation, ZeroPadding2D
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D
from keras.models import Sequential, Model
from keras.optimizers import RMSprop
from functools import partial

import tensorflow as tf
import keras.backend as K
from keras.backend.tensorflow_backend import set_session
import matplotlib.pyplot as plt
import sys
import numpy as np
import os
from keras.models import load_model
from keras.models import model_from_json

os.environ['CUDA_VISIBLE_DEVICES']='0'
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
# set_session(tf.Session(config=config))

img_rows = 28
img_cols = 28
channels = 1
img_shape = (img_rows, img_cols, channels)
latent_dim = 100

n_critic = 5#鉴别器训练5次，才训练一次生成器
optimizer = RMSprop(lr=0.00005)#定义优化函数

####################################################################
#在真实样本和生成样本间随机插值采样
class RandomWeightedAverage(_Merge):
    def _merge_function(self, inputs):
        alpha = K.random_uniform((32, 1, 1, 1))
        return (alpha * inputs[0]) + ((1 - alpha) * inputs[1])

def build_load_model():
    json_file = open('E:/DeepLearn/AutoEncodermodel/AutoEncoder_model.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)

    loaded_model.load_weights('E:/DeepLearn/AutoEncodermodel/AutoEncoder_model_weights.hdf5')
    print("load model from disk")

    loaded_model.compile(loss='binary_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    return loaded_model

def gradient_penalty_loss(y_true, y_pred, averaged_samples):
    """
    Computes gradient penalty based on prediction and weighted real / fake samples
    """
    gradients = K.gradients(y_pred, averaged_samples)[0]
    # compute the euclidean norm by squaring ...
    gradients_sqr = K.square(gradients)
    #   ... summing over the rows ...
    gradients_sqr_sum = K.sum(gradients_sqr,
                                  axis=np.arange(1, len(gradients_sqr.shape)))
    #   ... and sqrt
    gradient_l2_norm = K.sqrt(gradients_sqr_sum)
    # compute lambda * (1 - ||grad||)^2 still for each single sample
    gradient_penalty = K.square(1 - gradient_l2_norm)
    # return the mean as loss over all the batch samples
    return K.mean(gradient_penalty)


def wasserstein_loss(y_true, y_pred):
    return K.mean(y_true * y_pred)

def build_generator():

    model = Sequential()

    model.add(Dense(128 * 7 * 7, activation="relu", input_dim=latent_dim))
    model.add(Reshape((7, 7, 128)))
    model.add(UpSampling2D())
    model.add(Conv2D(128, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(UpSampling2D())
    model.add(Conv2D(64, kernel_size=4, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(Activation("relu"))
    model.add(Conv2D(channels, kernel_size=4, padding="same"))
    model.add(Activation("tanh"))

    model.summary()

    noise = Input(shape=(latent_dim,))
    img = model(noise)

    return Model(noise, img)

def build_critic():

    model = Sequential()

    model.add(Conv2D(16, kernel_size=3, strides=2, input_shape=img_shape, padding="same"))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(32, kernel_size=3, strides=2, padding="same"))
    model.add(ZeroPadding2D(padding=((0,1),(0,1))))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(64, kernel_size=3, strides=2, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Conv2D(128, kernel_size=3, strides=1, padding="same"))
    model.add(BatchNormalization(momentum=0.8))
    model.add(LeakyReLU(alpha=0.2))
    model.add(Dropout(0.25))
    model.add(Flatten())
    model.add(Dense(1))

    model.summary()

    img = Input(shape=img_shape)
    validity = model(img)

    return Model(img, validity)

# Build the generator and critic
generator = build_generator()
critic = build_critic()
#-------------------------------
# Construct Computational Graph
#       for the Critic
#-------------------------------
# Freeze generator's layers while training critic
generator.trainable = False

# Image input (real sample)
real_img = Input(shape=img_shape)

# Noise input
z_disc = Input(shape=(latent_dim,))
# Generate image based of noise (fake sample)
fake_img = generator(z_disc)

# 判断样本的真假
fake = critic(fake_img)
valid = critic(real_img)

# Construct weighted average between real and fake images
interpolated_img =RandomWeightedAverage()([real_img, fake_img])
# Determine validity of weighted sample
validity_interpolated = critic(interpolated_img)
####################################################################
# Use Python partial to provide loss function with additional
# 'averaged_samples' argument
#构建惩罚项损失函数
partial_gp_loss = partial(gradient_penalty_loss,
                          averaged_samples=interpolated_img)
partial_gp_loss.__name__ = 'gradient_penalty' # Keras requires function names

critic_model = Model(inputs=[real_img, z_disc],outputs=[valid, fake, validity_interpolated])
critic_model.compile(loss=[wasserstein_loss,#因为是三个输出值，所以有三个损失函数，最后的输出为三个损失值的和
                                              wasserstein_loss,
                                              partial_gp_loss],
                                        optimizer=optimizer,
                                        loss_weights=[1,1, 10])#根据权重衡量不同损失函数对输出的贡献
#-------------------------------
# Construct Computational Graph
#         for Generator
#-------------------------------

# For the generator we freeze the critic's layers
critic.trainable = False
generator.trainable = True

# Sampled noise for input to generator
z_gen = Input(shape=(100,))
# Generate images based of noise
img = generator(z_gen)
# Discriminator determines validity
valid = critic(img)
# Defines generator model
generator_model = Model(z_gen, valid)
generator_model.compile(loss=wasserstein_loss, optimizer=optimizer)




def train(epochs, batch_size, sample_interval=50):

    # Load the dataset
    (X_train, _), (_, _) = mnist.load_data("MNIST_data")

    # Rescale -1 to 1
    X_train = (X_train.astype(np.float32) - 127.5) / 127.5
    X_train = np.expand_dims(X_train, axis=3)
    #加载编码器模型
    encoder=build_load_model()
    X_train = np.reshape(X_train, (len(X_train), 28, 28, 1))
    X_train = encoder.predict(X_train)
    # 定义可视化操作
    sess = tf.Session()
    writer = tf.summary.FileWriter('/temp/data/logs', sess.graph)
    D_loss = tf.placeholder(tf.float32)
    G_loss = tf.placeholder(tf.float32)
    tf.summary.scalar("D_loss", D_loss)
    tf.summary.scalar("G_loss", G_loss)
    merged = tf.summary.merge_all()

    # Adversarial ground truths
    valid = -np.ones((batch_size, 1))
    fake = np.ones((batch_size, 1))
    dummy = np.zeros((batch_size, 1))  # Dummy gt for gradient penalty
    for epoch in range(epochs):

        for _ in range(n_critic):
            # ---------------------
            #  Train Discriminator
            # ---------------------
            # Select a random batch of images
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]
            # Sample generator input
            noise = np.random.normal(0, 1, (batch_size, latent_dim))
            # Train the critic
            d_loss = critic_model.train_on_batch([imgs, noise], [valid, fake, dummy])

        # ---------------------
        #  Train Generator
        # ---------------------
        g_loss = generator_model.train_on_batch(noise, valid)

        # Plot the progress
        print("%d [D loss: %f] [G loss: %f]" % (epoch, d_loss[0], g_loss))

        # If at save interval => save generated image samples
        if epoch % 500 == 0:
            sample_images(epoch)
        d_loss = (d_loss[0] + d_loss[1] + d_loss[2] + d_loss[3]) / 4
        # 将训练结果保存到可视化数据
        summary = sess.run(merged, feed_dict={G_loss: g_loss, D_loss: d_loss})
        writer.add_summary(summary, epoch)
    save_model()


def sample_images(epoch):
    r, c = 5, 5
    noise = np.random.normal(0, 1, (r * c, latent_dim))
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
    fig.savefig("E:/DeepLearn/myproject/mnist_%d.png" % epoch)
    plt.close()


def save_model():
    def save(model, model_name):
        model_path = "E:/DeepLearn/WGANModel/%s.json" % model_name
        weights_path = "E:/DeepLearn/WGANModel/%s_weights.hdf5" % model_name
        options = {"file_arch": model_path,
                   "file_weight": weights_path}
        json_string = model.to_json()
        open(options['file_arch'], 'w').write(json_string)
        model.save_weights(options['file_weight'])

    save(generator, "WGAN_model")

train(epochs=20000, batch_size=32, sample_interval=100)
