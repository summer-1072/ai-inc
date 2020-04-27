import numpy as np
import matplotlib.pyplot as plt


def mse(array1, array2):
    return np.sum((array1 - array2) ** 2)


def show(title, trajectory):
    plt.figure(figsize=(9, 6))
    plt.title(title)
    plt.plot(trajectory, linestyle="-.")
    plt.show()
