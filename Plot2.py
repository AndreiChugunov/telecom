from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal


Fdiscrete = 8e3
t = np.linspace(0, 1, int(Fdiscrete))
f = np.asarray([600, 800, 1000, 1200, 1400])
s3 = [[np.cos(2 * np.pi * fi * ti) for ti in t] for fi in f]
plt.figure(1)
for si in s3: plt.plot(si[0:100])
plt.grid()
plt.show()
