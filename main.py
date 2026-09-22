import torch
import numpy as np
import matplotlib.pyplot as plt

from generators import SignalGenerator
from model import DenoiseNet

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = DenoiseNet().to(device)
model.load_state_dict(torch.load("denoise_model.pth", map_location=device))
model.eval()

generator = SignalGenerator()

clean, noisy, t, f = generator.generate(duration=1.0)

frame = 128
hop = 64
window = torch.hann_window(frame).to(device)

total_samples = len(noisy)
output = torch.zeros(total_samples, device=device)

for i in range(0, 48000 - frame, hop):
    signal_chunk = noisy[i:i+frame]

    noisy_tensor = torch.from_numpy(signal_chunk.astype(np.float32)).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(noisy_tensor)

    pred_windowed = pred.squeeze() * window

    output[i:i+frame] += pred_windowed

denoised = output.cpu().numpy()

plt.figure(figsize=(12, 4))
plt.plot(t[:500], noisy[:500], label='Noisy', alpha=0.3, color='k')
plt.plot(t[:500], clean[:500], label='Clean', alpha=0.7, color='r')
plt.plot(t[:500], denoised.T[:500], label='Prediction', color='b')
plt.legend()
plt.title(f'Sine wave ({f:.2f} Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.tight_layout()
plt.show()