from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


Fs = 1e3
t = np.linspace(-40e-3, 40e-3, int(Fs))
T = 20e-3
A = 5

def Srect(t, width):
    return [int(-width / 2 <= ti < width / 2) for ti in t]

S = -A * np.asarray(Srect(t + T / 2, T)) + A * np.asarray(Srect(t - T / 2, T))
plt.figure()
plt.plot(t[0:len(S)], S)
plt.axis(ylim=[-6, 6])
plt.show()
