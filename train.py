import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from dataset import SignalDataset
from model import DenoiseNet

batch_size = 32
epochs = 10
lr = 0.001
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

train_dataset = SignalDataset()
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
model = DenoiseNet().to(device)
optimizer = optim.Adam(model.parameters(), lr=lr)
criterion = nn.MSELoss()

for epoch in range(epochs):
    model.train()
    running_loss = 0.0

    for x_batch, y_batch in train_loader:
        x_batch, y_batch = x_batch.to(device), y_batch.to(device)

        optimizer.zero_grad()
        y_pred = model(x_batch)
        loss = criterion(y_pred, y_batch)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader)
    print(f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.6f}")

torch.save(model.state_dict(), "denoise_model.pth")