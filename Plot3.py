from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

t = np.linspace(-2, 2, 1000)
T = 0.5
alpha = 10
A = 2

S = [A * np.exp(-alpha * ti) if ti >= 0 else 0 for ti in t]
plt.figure(2)
plt.plot(t, S)

S = [A if np.abs(ti) <= T / 2 else 0 for ti in t]
plt.plot(t, S)

S = [A * ti / T if 0 <= ti <= T else 0 for ti in t]
plt.plot(t, S)

plt.show()