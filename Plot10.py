from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def diric(x, n): return np.sin(n * x / 2) / (n * np.sin(x / 2))


x = np.linspace(0, 15, 1 / 0.01)
plt.figure()
plt.plot(x, diric(x, 7))
plt.figure()
plt.plot(x, diric(x, 8))
plt.show()

