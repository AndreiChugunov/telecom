from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


Fs = 1e3
t = np.linspace(-25e-3, 125e-3, int(Fs))
A = 5
T = 50e-3
t1 = 5e-3
plt.figure()
S = (signal.sawtooth(2 * np.pi * t / T, 1 - t1 / T) - 1) * A / 2
plt.plot(t, S)

Nfft = int(2 ** np.ceil(np.log2(len(S))))
sp = fft(S, Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fs
plt.figure()
plt.grid()
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
plt.axis([-10, 40, 0, 3000])

plt.show()