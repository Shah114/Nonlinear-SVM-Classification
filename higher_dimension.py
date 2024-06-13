# Importing modules
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

x1D = np.linspace(-4, 4, 9).reshape(-1, 1)
x2D = np.c_[x1D, x1D ** 2]
y = np.array([0, 0, 1, 1, 1, 1, 1, 0, 0])

# Plotting
plt.figure(figsize=(10, 3))

plt.subplot(121)
plt.grid(True)
plt.axhline(y=0, color='k')
plt.plot(x1D[:, 0][y==0], np.zeros(4), "bs")
plt.plot(x1D[:, 0][y==1], np.zeros(5), "g^")
plt.gca().get_yaxis().set_ticks([])
plt.xlabel("$x_1$")
plt.axis([-4.5, 4.5, -0.2, 0.2])

plt.subplot(122)
plt.grid(True)
plt.axhline(y=0, color='k')
plt.axvline(x=0, color='k')
plt.plot(x2D[:, 0][y==0], x2D[:, 1][y==0], "bs")
plt.plot(x2D[:, 0][y==1], x2D[:, 1][y==1], "g^")
plt.xlabel("$x_1$")
plt.ylabel("$x_2$", rotation=0)
plt.gca().get_yaxis().set_ticks([0, 4, 8, 12, 16])
plt.plot([-4.5, 4.5], [6.5, 6.5], "r--", linewidth=3)
plt.axis([-4.5, 4.5, -1, 17])

plt.subplots_adjust(right=1)

plt.show()