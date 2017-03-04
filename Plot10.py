from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

def diric(x, n):
    S = []
    for xi in x:
        if xi == 0:
            S.append(1)
        else:
            S.append(np.sin(n * xi / 2) / (n * np.sin(xi / 2)))
    return np.asarray(S)


x = np.linspace(0, 15, 1 / 0.01)
Fs = 1 / 0.01;
plt.figure()
plt.plot(x, diric(x, 7))
plt.figure()
Nfft = int(2 ** np.ceil(np.log2(len(diric(x, 7)))))
print(Nfft)
sp = fft(diric(x, 7), Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fs
plt.grid()
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
plt.figure()
plt.plot(x, diric(x, 8))
plt.figure()
Nfft = int(2 ** np.ceil(np.log2(len(diric(x, 8)))))
sp = fft(diric(x, 8), Nfft)
print(sp)
print(Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fs
plt.grid()
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
plt.show()

