import numpy as np

class SignalGenerator:
    def __init__(self):
        self.fs = 48000
        self.f_min = 50
        self.f_max = 1000
        self.phase_min = 0.0
        self.phase_max = 2.0
        self.std_min = 0.1
        self.std_max = 0.5

    def generate(self, duration=None, chunk=None):
        if duration is not None and chunk is not None:
            raise ValueError("Too many parameters specified.")

        if duration is not None:
            n_samples = round(duration * self.fs)
        elif chunk is not None:
            n_samples = int(chunk)
        else:
            raise ValueError("Parameter not specified.")

        t = np.arange(n_samples) / self.fs

        freq = np.random.uniform(self.f_min, self.f_max)
        std = np.random.uniform(self.std_min, self.std_max)
        phase = np.random.uniform(self.phase_min, self.phase_max) * np.pi

        clean_signal = np.sin(2 * np.pi * freq * t + phase)
        noise = np.random.normal(0, std, len(clean_signal))
        noisy_signal = clean_signal + noise

        return clean_signal, noisy_signal, t, freq

# np.random.randn() - torch.randn()
# np.random.normal() - torch.normal()
# np.random.uniform() - torch.rand()