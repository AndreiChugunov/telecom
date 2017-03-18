from scipy.fftpack import fft
import numpy as np
import time as time
import random
import matplotlib.pyplot as plt
from scipy import signal

sig = np.asarray([0, 0, 0, 1, 0, 1, 0, 1, 1, 1, 0, 0, 0, 0, 1, 0])
syncSig = np.asarray([1, 0, 1])

sig = sig + sig - 1
syncSig = syncSig + syncSig - 1
print('correlation')
print('the signal: ', sig)
print('the sync signal: ', syncSig)

correlation = np.asarray([])
t = np.arange(1000)
sum = 0
for ti in t:
    beforeTime = time.time()
    correlation = np.correlate(sig, syncSig)
    afterTime = time.time()
    sum = sum + afterTime - beforeTime
print('time needed for correlation: ', sum / len(t))

print('-----------------------------------------------------------------------------')
print('fast correlation')
sig = np.asarray([-1, -1, -1, 1, -1, 1, -1, 1, 1, 1, -1, -1, -1, -1, 1, -1, 0, 0])
syncSig = np.asarray([1, -1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
print('the signal: ', sig)
print('the sync signal: ', syncSig)
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
print('correlation result: ', correlation)
print('fast correlation result: ', finalCorrel[:14].real)

print('-----------------------------------------------------------------------------')
print('-----------------------------------------------------------------------------')
print('extended signal')
print('')
print('correlation')
s = []
size = np.arange(10000)
for si in size:
    s.append(random.randint(0, 1))
sig = np.asarray(s)
syncSig = np.asarray([1, 0, 1])

sig = sig + sig - 1
syncSig = syncSig + syncSig - 1

print('the signal: ', sig)
print('the sync signal: ', syncSig)

correlation = np.asarray([])
sum = 0
for ti in t:
    beforeTime = time.time()
    correlation = np.correlate(sig, syncSig)
    afterTime = time.time()
    sum = sum + afterTime - beforeTime
print('time needed for correlation in an extended signal: ', sum / len(t))


print('-----------------------------------------------------------------------------')
print('fast correlation')
sizeZeroListForSig = np.arange(len(syncSig) - 1)
sizeZeroListForSyncSig = np.arange(len(sig) - 1)
sig = list(sig)
for z in sizeZeroListForSig:
    sig.append(0)
syncSig = list(syncSig)
for zi in sizeZeroListForSyncSig:
    syncSig.append(0)
sig = np.asarray(sig)
syncSig = np.asarray(syncSig)

print('the signal: ', sig)
print('the sync signal: ', syncSig)

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
print('time needed for fast correlation in an extended signal: ', sum / len(t))
print('correlation in an extended signal result: ', correlation)
print(len(sig) - len(syncSig))
print('fast correlation in an extended signal result: ', finalCorrel[:9997 ].real)
