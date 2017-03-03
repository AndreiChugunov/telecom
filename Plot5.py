from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def Striang(t, width, skew=0):
    S = []
    for ti in t:
        if -width / 2 <= ti < width * skew / 2:
            S.append((2 * ti + width) / (width * (skew + 1)))
        elif width * skew / 2 <= ti < width / 2:
            S.append((2 * ti - width) / (width * (skew - 1)))
        elif np.abs(ti) > width / 2:
            S.append(0)
    return np.asarray(S)


Fs = 1e3
T1 = 20e-3
t = np.linspace(-50e-3, 50e-3, int(Fs))
A = 10

T2 = 60e-3
s = A * (T2 * np.asarray(Striang(t, T2, 0)) - T1 * np.asarray(Striang(t, T1, 0))) / (T2 - T1)
plt.figure()
plt.plot(t[0:len(s)], s)
plt.show()