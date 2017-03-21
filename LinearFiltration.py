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
sig = np.asarray(sig)
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

filteredSig = np.asarray(noiseSig)
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

noiseSignal = A / 4 * np.cos(20 * np.pi * f0 * t + phi) + sig
pureSig = signal.medfilt(noiseSignal, 85)
plt.figure()
plt.plot(t, pureSig)

Nfft = int(2 ** np.ceil(np.log2(len(pureSig))))
sp = fft(pureSig, Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fdiscrete
plt.figure()
plt.grid()
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
plt.axis([0, 250, 0, 4000])

cutoff = 0.1
noiseSig =  A / 4 * np.cos(20 * np.pi * f0 * t + phi) + sig
b, a = signal.butter(4, cutoff/(f0/2), btype='low', analog=False, output='ba')
y = signal.lfilter(b, a, noiseSig)
# Get the filter coefficients so we can check its frequency response.
plt.figure()
plt.plot(t, y)

Nfft = int(2 ** np.ceil(np.log2(len(y))))
sp = fft(y, Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fdiscrete
plt.figure()
plt.grid()
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
plt.axis([0, 250, 0, 4000])

plt.show()