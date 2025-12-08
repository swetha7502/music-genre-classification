# import torch
# from torch.utils.data import DataLoader
# from dataset import GenreDataset
# from model import GenreCNN
# import torch.nn as nn
# import torch.optim as optim

# dataset = GenreDataset("data/mels/")
# train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

# device = "cuda" if torch.cuda.is_available() else "cpu"

# model = GenreCNN(num_classes=10).to(device)
# criterion = nn.CrossEntropyLoss()
# optimizer = optim.Adam(model.parameters(), lr=0.0003)

# EPOCHS = 15

# for epoch in range(EPOCHS):
#     running_loss = 0
#     correct = 0
#     total = 0

#     for mel, label in train_loader:
#         mel, label = mel.to(device), label.to(device)

#         optimizer.zero_grad()
#         outputs = model(mel)
#         loss = criterion(outputs, label)
#         loss.backward()
#         optimizer.step()

#         running_loss += loss.item()
#         _, predicted = torch.max(outputs.data, 1)
#         total += label.size(0)
#         correct += (predicted == label).sum().item()

#     print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.3f}, Acc: {100*correct/total:.2f}%")

# torch.save(model.state_dict(), "genre_cnn.pth")

import torch
from torch.utils.data import DataLoader
from dataset import GenreDataset
from model import GenreCNN
import torch.nn as nn
import torch.optim as optim

dataset = GenreDataset("data/mels/")
train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

device = "cuda" if torch.cuda.is_available() else "cpu"

model = GenreCNN(num_classes=10).to(device)

# Fix: Get the actual flattened size from a sample batch
sample_mel, _ = next(iter(train_loader))
sample_mel = sample_mel.to(device)
with torch.no_grad():
    sample_output = model.conv_layers(sample_mel)
    flattened_size = sample_output.view(sample_output.size(0), -1).size(1)

# Reinitialize model with correct input size
model = GenreCNN(num_classes=10, input_size=flattened_size).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.0003)

EPOCHS = 15

for epoch in range(EPOCHS):
    running_loss = 0
    correct = 0
    total = 0

    for mel, label in train_loader:
        mel, label = mel.to(device), label.to(device)

        optimizer.zero_grad()
        outputs = model(mel)
        loss = criterion(outputs, label)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += label.size(0)
        correct += (predicted == label).sum().item()

    print(f"Epoch {epoch+1}/{EPOCHS}, Loss: {running_loss:.3f}, Acc: {100*correct/total:.2f}%")

torch.save(model.state_dict(), "genre_cnn.pth")