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

clean, noisy, t, f = generator.generate()
noisy_tensor = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).to(device)

with torch.no_grad():
    pred_tensor = model(noisy_tensor)
    pred = pred_tensor.cpu().detach().numpy()

plt.figure(figsize=(12, 4))
plt.plot(t, noisy, label='Noisy', alpha=0.3, color='k')
plt.plot(t, clean, label='Clean', alpha=0.7, color='r')
plt.plot(t, pred.T, label='Prediction', color='b')
plt.legend()
plt.title(f'Sine wave ({f} Hz)')
plt.xlabel('Time [s]')
plt.ylabel('Amplitude')
plt.grid()
plt.tight_layout()
plt.show()