from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


Fdiscrete = 32e3
t = np.linspace(0, 1, int(Fdiscrete))
f = np.asarray([600, 800, 1000])
s3 = [[np.cos(2 * np.pi * fi * ti) for ti in t] for fi in f]
plt.figure(1)
plt.grid()
for si in s3: plt.plot(si[0:100])
plt.figure()

for si in s3:
    Nfft = int(2 ** np.ceil(np.log2(len(si))))
    sp = fft(si, Nfft)
    sp_dB = 20 * np.log10(np.abs(sp))
    f = np.arange(0, Nfft - 1) / Nfft * Fdiscrete
    plt.grid()
    plt.plot(f[:int(Nfft / 2)], np.abs(sp[:int(Nfft / 2)]))

plt.axis([0, 4000, 0, 200])
plt.show()
