import numpy as np
import matplotlib.pyplot as plt

x = np.arange(0, np.pi * 2, 0.1)
y = np.sin(x)

plt.plot(x, y)
plt.show()
