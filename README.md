# 🌊 CNN Denoising Autoencoder

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)

<p align="center">
  <img src="readme_files/example.png" alt="Denoising Example" />
</p>

## 📝 Overview
A real-time signal denoising pipeline built in **PyTorch** using a 1D Convolutional Autoencoder (`nn.Conv1d` / `nn.ConvTranspose1d`). 

The model is trained purely on **synthetic on-the-fly generated signals** sampled at a production-grade rate of **48 kHz**, with chunk-based windowing (128 samples per block) and reconstruction via **Overlap-Add (OLA)** using Hann windowing to prevent boundary artifacts in continuous streaming.

## 🚀 Key Features
- **1D Convolutional Autoencoder**: Symmetrical encoder-decoder architecture that compresses the receptive field to filter high-frequency noise and reconstruct smooth sinusoidal features.
- **Dynamic On-the-Fly Dataset**: Generates endless synthetic training batches with randomized frequencies, initial phases, and Gaussian noise levels (`torch.utils.data.Dataset`), preventing overfitting without saving large files to disk.
- **Real-Time Streaming Ready**: Implements frame-based processing (128 samples, ~2.67 ms latency) with a 50% hop-size **Overlap-Add (OLA)** routine and edge padding/windowing for continuous streams.
- **GPU Accelerated**: Seamless end-to-end execution pipeline supporting both CUDA and CPU inference.

## 🛠️ Tech Stack
- **Language**: `Python 3.10+`
- **Libraries**:
  - `PyTorch` – Deep learning framework and tensor operations
  - `NumPy` – Signal synthesis and vector math
  - `Matplotlib` – Waveform visualization and spectral inspection