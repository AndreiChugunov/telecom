from scipy.fftpack import fft
import numpy as np
import time as time
import matplotlib.pyplot as plt
from scipy import signal

sig = np.asarray([0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0])
syncSig = np.asarray([1, 0, 1])

sig = sig + sig - 1
syncSig = syncSig + syncSig - 1

print(sig)
print(syncSig)

correlation = np.asarray([])
t = np.arange(1e6)
sum = 0
for ti in t:
    beforeTime = time.time()
    correlation = np.correlate(sig, syncSig)
    afterTime = time.time()
    sum = sum + afterTime - beforeTime
print('time needed for correlation: ', sum / len(t))

sig = np.asarray([-1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 0, 0])
syncSig = np.asarray([1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
sum = 0
finalCorrel = np.asarray([])
for di in t:
    beforeTime = time.time()
    conjY = np.conjugate(np.fft.fft(syncSig))
    y1Fft = np.fft.fft(sig)
    multiplication = conjY * y1Fft
    finalCorrel = np.fft.ifft(multiplication)
    afterTime = time.time()
    sum = sum + afterTime - beforeTime
print('time needed for fast correlation: ', sum / len(t))
print(correlation)
print(finalCorrel[:14].real)

