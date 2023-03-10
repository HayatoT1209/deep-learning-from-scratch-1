import numpy as np
import matplotlib.pyplot as plt
from step_function import step_function
from sigmoid import sigmoid


x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
plt.plot(x, y1, label="step")
plt.plot(x, y2, "k--", label="sigmoid")
plt.ylim(-0.1, 1.1)
plt.legend()
plt.title("Step and Sigmoid Functions")
plt.show()
