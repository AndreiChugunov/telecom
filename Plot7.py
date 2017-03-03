from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Fs = 16000
t = np.arange(-10e-3, 10e-3, 1 / Fs)
Fc = 4000
bw = 0.1
bwr = -20
s = signal.gausspulse(t, Fc, bw, bwr)
Nfft = int(2 ** np.ceil(np.log2(len(s))))
sp = fft(s, Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fs
sp_max_dp = 20 * np.log10(np.max(np.abs(sp)))
edges = Fc * np.asarray([1 - bw / 2, 1 + bw / 2])

plt.figure()
plt.grid()
plt.plot(t, s)

plt.figure()
plt.grid()
plt.plot(f[:int(Nfft / 2)], sp_dB[:int(Nfft / 2)])
plt.plot(edges, sp_max_dp * np.asarray([1, 1]) + bwr, "o")

plt.show()