from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Fs = 8e3
t = np.linspace(0, 1, int(Fs))
f0 = 1e3
t1 = 1
f1 = 2e3
s1 = signal.chirp(t, f0, t1, f1, 'linear')
s2 = signal.chirp(t, f0, t1, f1, 'quadratic')
s3 = signal.chirp(t, f0, t1, f1, 'logarithmic')
plt.figure()
plt.specgram(np.asarray(s1), None, int(Fs))
plt.figure()
plt.specgram(np.asarray(s2), None, int(Fs))
plt.figure()
plt.specgram(np.asarray(s3), None, int(Fs))
plt.show()