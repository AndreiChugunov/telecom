from scipy.fftpack import fft
import numpy as np
import time as time
import random
import matplotlib.pyplot as plt
from scipy import signal


def spectrum(noiseSig, Fdiscrete):
    Nfft = int(2 ** np.ceil(np.log2(len(noiseSig))))
    sp = fft(noiseSig, Nfft)
    f = np.arange(0, Nfft - 1) / Nfft * Fdiscrete
    plt.figure()
    plt.grid()
    plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
    plt.axis([0, 250, 0, 4000])

Fdiscrete = int(4e3)
t = np.linspace(0, 1, int(Fdiscrete))
A = 2
f0 = 10
phi = np.pi / 4
sig = A * np.cos(2 * np.pi * f0 * t + phi)
mean = 0
std = 1
samples = 5 * np.random.normal(mean, std, size = Fdiscrete)

noiseSig = sig + samples
plt.figure()
plt.plot(t, sig)
plt.figure()
plt.plot(t, noiseSig)
spectrum(noiseSig, Fdiscrete)


Length = 300
a = signal.firwin(Length, cutoff = f0, window="hamming", nyq = Fdiscrete / 2)
newSig = signal.lfilter(a, 1, noiseSig)

plt.figure()
plt.plot(t, newSig)
spectrum(newSig, Fdiscrete)
plt.show()