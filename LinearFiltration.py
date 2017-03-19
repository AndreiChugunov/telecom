from scipy.fftpack import fft
import numpy as np
import time as time
import random
import matplotlib.pyplot as plt
from scipy import signal


Fdiscrete = 4e3
t = np.linspace(0, 1, int(Fdiscrete))
A = 2
f0 = 10
phi = np.pi / 4
sig = A * np.cos(2 * np.pi * f0 * t + phi)
noiseSig = A / 4 * np.cos(20 * np.pi * f0 * t + phi) + sig

Nfft = int(2 ** np.ceil(np.log2(len(sig))))
sp = fft(sig, Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fdiscrete
plt.figure()
plt.grid()
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
plt.axis([0, 250, 0, 4000])

Nfft = int(2 ** np.ceil(np.log2(len(noiseSig))))
sp = fft(noiseSig, Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fdiscrete
plt.figure()
plt.grid()
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
plt.axis([0, 250, 0, 4000])

plt.figure()
plt.plot(t, sig)
plt.figure()
plt.plot(t, noiseSig)

filteredSig = noiseSig
window = np.arange(50)
for si in (np.arange(len(noiseSig) - len(window))):
    sum = 0
    for i in window:
        sum = sum + noiseSig[si + i]
    filteredSig[si] = sum / len(window)
lastPart = np.arange(len(noiseSig) - len(window), len(noiseSig))
for si in lastPart:
    sum = 0
    for i in np.arange((len(noiseSig) - si)):
        sum = sum + noiseSig[si + i]
    for iti in np.arange(len(window) - (len(noiseSig) - si)):
        sum = sum + noiseSig[len(noiseSig) - 1]
    filteredSig[si] = sum / len(window)
plt.figure()
plt.plot(t, filteredSig)

Nfft = int(2 ** np.ceil(np.log2(len(filteredSig))))
sp = fft(filteredSig, Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fdiscrete
plt.figure()
plt.grid()
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
plt.axis([0, 250, 0, 4000])

windowRange = np.arange(9)
mas = []
for si in np.arange(int(len(windowRange) / 2)):
    for di in np.arange(int(len(windowRange) / 2) - si):
        mas.append(noiseSig[0])

for si in np.arange(int(len(windowRange) / 2) + 1, len(noiseSig) - int(len(windowRange) / 2)):
    sum = sum + 1;

plt.show()