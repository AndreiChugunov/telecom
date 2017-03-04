from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

t = np.linspace(-2, 2, 10000)
T = 0.5
alpha = 10
A = 2

S1 = [A * np.exp(-alpha * ti) if ti >= 0 else 0 for ti in t]
plt.figure(2)
plt.plot(t, S1)

S2 = [A if np.abs(ti) <= T / 2 else 0 for ti in t]
plt.plot(t, S2)

S3 = [A * ti / T if 0 <= ti <= T else 0 for ti in t]
plt.plot(t, S3)


Nfft = int(2 ** np.ceil(np.log2(len(S1))))
sp = fft(S1)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * 10000
plt.figure()
plt.grid()
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
plt.axis([-10, 200, 0, 500])
#plt.plot(sp)

Nfft = int(2 ** np.ceil(np.log2(len(S2))))
sp = fft(S2)
print(sp)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * 10000
plt.figure()
plt.grid()
#plt.plot(sp)
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
plt.axis([-10, 200, 0, 2500])

Nfft = int(2 ** np.ceil(np.log2(len(S3))))
sp = fft(S3)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * 10000
plt.figure()
plt.grid()
#plt.plot(sp)
plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))
plt.axis([-10, 200, 0, 500])


plt.show()