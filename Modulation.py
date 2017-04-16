from scipy.fftpack import fft
import scipy.integrate as integrate
import numpy as np
import sympy as sp
import matplotlib.pyplot as plt
from scipy import signal


Fdiscrete = int(4e3)


def spectrum(noiseSig):
    Nfft = int(2 ** np.ceil(np.log2(len(noiseSig))))
    sp = fft(noiseSig, Nfft)
    f = np.arange(0, Nfft - 1) / Nfft * Fdiscrete
    plt.figure()
    plt.grid()
    plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
   # plt.axis([0, 250, 0, 4000])


def plotting(t, sig):
    plt.figure()
    plt.grid()
    plt.plot(t, sig)


t = np.linspace(0, 1, int(Fdiscrete))
A = 1
f0 = 10
Fc = 100
m = np.arange(0.5, 1.1, 0.5)
sig = A * np.cos(2 * np.pi * f0 * t)
plotting(t, sig)
spectrum(sig.real)
for mi in m:
    modulatedSig = (1 + mi * sig) * np.cos(2 * np.pi * Fc * t)
    plotting(t, modulatedSig)
    spectrum(modulatedSig.real)
for mi in m:
    killTheCarrier = mi * sig * np.cos(2 * np.pi * Fc * t)
    plotting(t, killTheCarrier)
    spectrum(killTheCarrier.real)
#ssbModulated = sig * np.cos(2 * np.pi * Fc * t) + (0.5 * A / 2) * np.cos(2 * np.pi * (Fc + f0) * t)
#ssbModulated = sig * np.cos(2 * np.pi * Fc * t) - signal.hilbert(sig).real * np.sin(2 * np.pi * Fc * t)
ssbModulated = A * 0.5 / 2 * np.cos(2 * np.pi * (Fc - f0) * t) + 0.5 * np.cos(2 * np.pi * Fc * t)
plotting(t, ssbModulated)
spectrum(ssbModulated.real)
demodulatedSig = ssbModulated * np.cos(2 * np.pi * Fc * t)
b, a = signal.butter(5, Fc/Fdiscrete*2, btype='lowpass', analog=False, output='ba')
z = signal.filtfilt(b, a, demodulatedSig)
plotting(t, z)
spectrum(z)
demodulatedSig = modulatedSig * np.cos(2 * np.pi * Fc * t)
b, a = signal.butter(5, 2 * Fc/ np.pi / Fdiscrete, btype='lowpass', analog=False, output='ba')
z = signal.filtfilt(b, a, demodulatedSig)
plotting(t,z)
spectrum(z)
demodulatedSig = killTheCarrier * np.cos(2 * np.pi * Fc * t)
b, a = signal.butter(5, Fc/Fdiscrete*2, btype='lowpass', analog=False, output='ba')
z = signal.filtfilt(b, a, demodulatedSig)
plotting(t,z)
spectrum(z)

mi = np.arange(0, 10, 0.1)
KPD = mi**2 / (mi**2 + 2)
plotting(mi, KPD)

phaseModulated = np.cos(2 * np.pi * Fc * t + np.pi * np.sin(2 * np.pi * f0 * t))
plotting(t, phaseModulated)
spectrum(phaseModulated)
z_pm = np.unwrap(np.angle(signal.hilbert(phaseModulated))) - 2 * np.pi * Fc * t
plotting(t, z_pm)
spectrum(z_pm)


frequencyModulated = np.cos(2 * np.pi * Fc * t + np.pi * np.sin(2 * np.pi * f0 * t))
plotting(t, frequencyModulated)
spectrum(frequencyModulated)
z_fm = np.diff(np.unwrap(np.angle(signal.hilbert(frequencyModulated)))) * Fdiscrete - 2 * np.pi * Fc
plotting(t[0:len(t) - 1], z_fm)
spectrum(z_fm)

plt.show()
