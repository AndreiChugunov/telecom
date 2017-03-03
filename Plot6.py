from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def Srect(t, width):
    return [int(-width / 2 <= ti < width / 2) for ti in t]

Fs = 1e3
t = np.linspace(-0.1, 0.1, int(Fs))
f0 = 10
T = 1 / f0
s = np.asarray(Srect(t, T)) * np.cos(2 * np.pi * f0 * t)
f = np.linspace(-50, 50, 100)
sp = T / 2 * (np.sinc((f - f0) * T) + np.sinc((f + f0) * T))
plt.figure()
plt.plot(t[0:len(s)], s)
plt.figure()
plt.plot(f[0:len(sp)], abs(sp))
plt.show()