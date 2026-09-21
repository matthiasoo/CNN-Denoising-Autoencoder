import numpy as np
import matplotlib.pyplot as plt

fs = 48000
channels = 1
bs = 128

t = np.arange(bs) / fs
freq = np.random.uniform(0.5, 1000)
std = np.random.uniform(0.1, 10)
phase = np.random.uniform(0, 2 * np.pi)

clean_signal = np.sin(2 * np.pi * freq * t + phase)
noisy_signal = clean_signal + np.random.normal(0, std, len(clean_signal))

plt.figure(figsize=(12, 4))
plt.plot(t, noisy_signal, label='Noisy', alpha=0.3, color='k')
plt.plot(t, clean_signal, label='Clean', color='r')
plt.legend()
plt.title(f'Sine wave ({freq} Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.tight_layout()
plt.show()

# np.random.randn() - torch.randn()
# np.random.normal() - torch.normal()
# np.random.uniform() - torch.rand()