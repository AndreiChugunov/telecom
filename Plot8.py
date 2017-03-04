from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Fs = 1e3
t = np.linspace(-10e-3, 50e-3, int(Fs))
A = 3
f0 = 50
tau = 5e-3
S = (signal.square(2 * np.pi * t * f0, f0 * tau) + 1) * A / 2
plt.figure()
plt.plot(t, S)



Nfft = int(2 ** np.ceil(np.log2(len(S))))
sp = fft(S, Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fs
plt.figure()
plt.grid()
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))

plt.show()