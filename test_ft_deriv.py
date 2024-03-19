import numpy as np

PI = np.pi
f = lambda x: np.sin(2 * PI * x)
fp = lambda x: 2 * PI * np.cos(2 * PI * x)
N = 65

x = np.linspace(0, 1, N)

y = f(x)
y_FT = np.fft.fft(y)
yp_FT = y_FT * 1j * 2 * PI * np.fft.fftfreq(N)
yp = np.fft.ifft(yp_FT)

import matplotlib.pyplot as plt

plt.plot(x, y, label="y")
plt.plot(x, y, label="f(x)")
plt.plot(x, fp(x), label="f'(x)")
plt.plot(x, yp, label="approx deriv")
plt.legend()
plt.tight_layout()
plt.show()
