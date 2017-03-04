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

plt.figure()
Nfft = int(2 ** np.ceil(np.log2(len(S))))
sp = fft(S, Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fs
plt.grid()
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
plt.axis([-10, 500, 0, 1500])

plt.show()
