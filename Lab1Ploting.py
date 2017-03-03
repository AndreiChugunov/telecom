from scipy.fftpack import fft
import numpy as np
import matplotlib.pyplot as plt
from scipy import signal

Fdiscrete = 8e3
t = np.linspace(0, 1, int(Fdiscrete))
A = 2
f0 = 1e3
phi = np.pi / 4
s1 = A * np.cos(2 * np.pi * f0 * t + phi)
alpha = 1000
s2 = np.exp(-alpha * t) * s1
plt.figure(0)
plt.subplot(2, 2, 1)
plt.plot(s2[0:100])
plt.grid()
plt.subplot(2, 2, 2)
plt.stem(s2[0:100])
plt.grid()
plt.subplot(2, 2, 3)
plt.plot(s2[0:100], '.')
plt.grid()
plt.subplot(2, 2, 4)
plt.step(t[0:100], s2[0:100])
plt.grid()

f = np.asarray([600, 800, 1000, 1200, 1400])
s3 = [[np.cos(2 * np.pi * fi * ti) for ti in t] for fi in f]
plt.figure(1)
for si in s3: plt.plot(si[0:100])
plt.grid()

t = np.linspace(-2, 2, 1000)
T = 0.5
alpha = 10

S = [A * np.exp(-alpha * ti) if ti >= 0 else 0 for ti in t]
plt.figure(2)
plt.plot(t, S)

S = [A if np.abs(ti) <= T / 2 else 0 for ti in t]
plt.plot(t, S)

S = [A * ti / T if 0 <= ti <= T else 0 for ti in t]
plt.plot(t, S)

Fs = 1e3
t = np.linspace(-40e-3, 40e-3, int(Fs))
T = 20e-3
A = 5


def Srect(t, width):
    return [int(-width / 2 <= ti < width / 2) for ti in t]


# S = -A * signal.square(t + T / 2, T) + A * signal.square(t - T / 2, T)
S = -A * np.asarray(Srect(t + T / 2, T)) + A * np.asarray(Srect(t - T / 2, T))
plt.figure()
plt.plot(t[0:len(S)], S)
plt.axis(ylim=[-6, 6])


def Striang(t, width, skew=0):
    S = []
    for ti in t:
        if -width / 2 <= ti < width * skew / 2:
            S.append((2 * ti + width) / (width * (skew + 1)))
        elif width * skew / 2 <= ti < width / 2:
            S.append((2 * ti - width) / (width * (skew - 1)))
        elif np.abs(ti) > width / 2:
            S.append(0)
    return np.asarray(S)


Fs = 1e3
T1 = 20e-3
t = np.linspace(-50e-3, 50e-3, int(Fs))
A = 10

T2 = 60e-3
s = A * (T2 * np.asarray(Striang(t, T2, 0)) - T1 * np.asarray(Striang(t, T1, 0))) / (T2 - T1)
plt.figure()
plt.plot(t[0:len(s)], s)

Fs = 1e3
t = np.linspace(-0.1, 0.1, int(Fs))
f0 = 10
T = 1 / f0
s = np.asarray(Srect(t, T)) * np.cos(2 * np.pi * f0 * t)
f = np.linspace(-50, 50, 100)
sp = T / 2 * (np.sinc((f - f0) * T) + np.sinc((f + f0) * T))
plt.figure()
plt.plot(t[0:len(s)], s)
plt.figure()
plt.plot(f[0:len(sp)], abs(sp))

Fs = 16000
t = np.arange(-10e-3, 10e-3, 1 / Fs)
Fc = 4000
bw = 0.1
bwr = -20
s = signal.gausspulse(t, Fc, bw, bwr)
Nfft = int(2 ** np.ceil(np.log2(len(s))))
sp = fft(s, Nfft)
sp_dB = 20 * np.log10(np.abs(sp))
f = np.arange(0, Nfft - 1) / Nfft * Fs
sp_max_dp = 20 * np.log10(np.max(np.abs(sp)))
edges = Fc * np.asarray([1 - bw / 2, 1 + bw / 2])

plt.figure()
plt.grid()
plt.plot(t, s)

plt.figure()
plt.grid()
plt.plot(f[:int(Nfft / 2)], sp_dB[:int(Nfft / 2)])
plt.plot(edges, sp_max_dp * np.asarray([1, 1]) + bwr, "o")

Fs = 1e3
t = np.linspace(-10e-3, 50e-3, int(Fs))
A = 3
f0 = 50
tau = 5e-3
S = (signal.square(2 * np.pi * t * f0, f0 * tau) + 1) * A / 2
plt.figure()
plt.plot(t, S)

Fs = 1e3
t = np.linspace(-25e-3, 125e-3, int(Fs))
A = 5
T = 50e-3
t1 = 5e-3
plt.figure()
S = (signal.sawtooth(2 * np.pi * t / T, 1 - t1 / T) - 1) * A / 2
plt.plot(t, S)


def diric(x, n): return np.sin(n * x / 2) / (n * np.sin(x / 2))


x = np.linspace(0, 15, 1 / 0.01)
plt.figure()
plt.plot(x, diric(x, 7))
plt.figure()
plt.plot(x, diric(x, 8))

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

def pulstran(t, d, a, foo, *args, **kwargs):
    assert len(a) == len(d)
    acc = np.zeros(len(t))
    for di, ai in zip(d, a):
        acc += ai * foo(t - di, *args, **kwargs)
    return acc


Fs = 1e3
t = np.arange(0, 0.5, 1 / Fs)
tau = 20e-3
d = np.array([20, 80, 160, 260, 380]) * 1e-3
a = 0.8 ** np.arange(0, 5)
y = pulstran(t, d, a, Striang, tau)
plt.figure()
plt.grid()
plt.plot(t, y)


Fs0 = 400
tau = 60e-3
t0 = np.linspace(0, tau, Fs0)
Fs = 1e3
t = np.linspace(0, 0.5, int(Fs))
s0 = np.asarray(np.sin(np.pi * t0 / tau)) ** 2
a = np.linspace(0, 5, 6)
Am = np.asarray([0.6 ** ai for ai in a])
d = np.linspace(1, 6, 6) * 64e-3

y = pulstran(t, d, Am, s0, tau)
plt.figure()
plt.grid()
plt.plot(t, y)
plt.show()



# n = np.linspace(0, tau, Fs0)
# n1 = []
# y1 = []
# plt.figure()
# for di in d:
#     n = (np.asarray(n) + tau)
#     n1 = np.concatenate((np.asarray(n1), np.asarray(n)))
#     print(di)
#     y1 = np.concatenate((np.asarray(y1), s0 * Am[di - 1]))
# plt.plot(n1, y1)
# plt.show()
