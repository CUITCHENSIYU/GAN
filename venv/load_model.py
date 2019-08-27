from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import load_model
from keras.models import model_from_json

(x_train, _), (_, _) = mnist.load_data("MNIST_data")
#数据预处理
x_train = x_train.astype('float32') / 255.
x_train = np.reshape(x_train, (len(x_train), 28, 28, 1))

json_file=open('E:/DeepLearn/AutoEncodermodel/AutoEncoder_model.json','r')
loaded_model_json=json_file.read()
json_file.close()
loaded_model=model_from_json(loaded_model_json)

loaded_model.load_weights('E:/DeepLearn/AutoEncodermodel/AutoEncoder_model_weights.hdf5')
print("load model from disk")

loaded_model.compile(loss='binary_crossentropy',optimizer='adadelta',metrics=['accuracy'])
gen_imgs=loaded_model.predict(x_train)

# Rescale images 0 - 1
gen_imgs = 0.5 * gen_imgs + 0.5

fig, axs = plt.subplots(5, 5)
cnt = 0
for i in range(5):
    for j in range(5):
        axs[i, j].imshow(gen_imgs[cnt, :, :, 0], cmap='gray')
        axs[i, j].axis('off')
        cnt += 1
fig.savefig("E:/DeepLearn/myproject/cod_.png")
plt.close()
print(x_train.shape)