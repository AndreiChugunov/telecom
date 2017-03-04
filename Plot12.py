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

def pulstran(t, d, a, foo, *args, **kwargs):
    assert len(a) == len(d)
    acc = np.zeros(len(t))
    for di, ai in zip(d, a):
        acc += ai * foo(t - di, *args, **kwargs)
    return acc


Fs = 1e3
t = np.arange(0, 0.5, 1 / Fs)
tau = 20e-3
d = np.array([20, 80, 160, 260, 380]) * 1e-3
a = 0.8 ** np.arange(0, 5)
y = pulstran(t, d, a, Striang, tau)
plt.figure()
plt.grid()
plt.plot(t, y)

Nfft = int(2 ** np.ceil(np.log2(len(y))))
sp = fft(y, Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fs
plt.figure()
plt.grid()
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))


plt.show()