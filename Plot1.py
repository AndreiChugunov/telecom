from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Fdiscrete = 8e3
t = np.linspace(0, 1, int(Fdiscrete))
A = 2
f0 = 1e3
phi = np.pi / 4
s1 = A * np.cos(2 * np.pi * f0 * t + phi)
alpha = 1000
s2 = np.exp(-alpha * t) * s1
plt.figure(0)
plt.subplot(2, 2, 1)
plt.plot(s2[0:100])
plt.grid()
plt.subplot(2, 2, 2)
plt.stem(s2[0:100])
plt.grid()
plt.subplot(2, 2, 3)
plt.plot(s2[0:100], '.')
plt.grid()
plt.subplot(2, 2, 4)
plt.step(t[0:100], s2[0:100])
plt.grid()
plt.show()