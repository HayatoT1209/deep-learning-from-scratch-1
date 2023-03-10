import sys, os
sys.path.append(os.pardir)
import numpy as np
import pickle
from dataset.mnist import load_mnist
from sigmoid import sigmoid
from softmax import softmax
import time


def get_data():
    (x_train, t_train), (x_test, t_test) = \
        load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test


def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)

    return network


def predict(network, x):
    # The network is a 3-stage network with 50 neurons on the first hidden layer,
    # and 100 neurons on the second hidden layer
    W1, W2, W3 = network['W1'], network['W2'],  network['W3']
    b1, b2, b3 = network['b1'], network['b2'],  network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data()
network = init_network()

# The np library is designed to compute calculation of larger matrix size faster.
# Hence, the use of batch will make the speed of prediction faster.
batch_size = 100
accuracy_cnt = 0

start_time = time.time()

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis=1)

    accuracy_cnt += np.sum(p == t[i:i+batch_size])

end_time = time.time()

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# Using batch speeds up the process by a factor of over 10
print("Total time: " + str(end_time - start_time) + " seconds")