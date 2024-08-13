import numpy as np

s = 16
center = s // 2
freq = 8

start = (s - min(s, freq)) // 2
end = -offset or None

x = np.linspace(0, 1, s, endpoint=False)
p2x = 2 * np.pi * x
print(x)
# activate frequencies 2, 4, 8
y = 4 * np.cos(4 * p2x)
y_ft = np.fft.fft(y, norm="forward")
y_ft = np.fft.fftshift(y_ft)
z_ft = np.zeros_like(y_ft)
z_ft[start:end] = y_ft[start:end]
print("zz", np.round(z_ft[start:end]))

print(y)
print("comp", np.round(y_ft), np.round(z_ft))
print("freq_diff", np.round(y_ft - z_ft))

print("freqs_kept", np.fft.fftshift(np.fft.fftfreq(s))[start:end])
print("freqs", np.fft.fftshift(np.fft.fftfreq(s)))


z_ft = np.fft.ifftshift(z_ft)
z = np.fft.ifft(z_ft, norm="forward")

# print(y, z)
