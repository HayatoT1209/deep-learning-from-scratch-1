# coding: utf-8
import sys, os
sys.path.append(os.pardir)
import numpy as np
import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from PIL import Image


def img_show(image):
    pil_img = Image.fromarray(np.uint8(image))
    pil_img.show()  # this method seems to be not working

    plt.imshow(pil_img)
    plt.show()


(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)
