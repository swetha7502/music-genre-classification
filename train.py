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
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split

from dataset import GenreDataset
from model import GenreCNN


def spec_augment(
    batch,
    freq_masks=2,
    time_masks=2,
    freq_mask_param=8,
    time_mask_param=20,
    apply_prob=0.5,
):
    """Lightweight SpecAugment with optional probability and capped mask sizes."""
    b, _, freq_max, time_max = batch.shape
    if torch.rand(1).item() > apply_prob:
        return batch
    for i in range(b):
        # Frequency masks
        for _ in range(freq_masks):
            if freq_max <= 1:
                continue
            width = torch.randint(0, min(freq_mask_param, freq_max) + 1, (1,)).item()
            if width == 0:
                continue
            start = torch.randint(0, max(1, freq_max - width + 1), (1,)).item()
            batch[i, 0, start : start + width, :] = 0
        # Time masks
        for _ in range(time_masks):
            if time_max <= 1:
                continue
            width = torch.randint(0, min(time_mask_param, time_max) + 1, (1,)).item()
            if width == 0:
                continue
            start = torch.randint(0, max(1, time_max - width + 1), (1,)).item()
            batch[i, 0, :, start : start + width] = 0
    return batch


def compute_class_weights(dataset):
    counts = torch.zeros(len(dataset.genres), dtype=torch.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    # Inverse frequency, normalized to mean 1 to keep loss scale stable
    weights = counts.mean() / counts
    return weights


def evaluate(model, loader, criterion, device):
    model.eval()
    loss_sum = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for mel, label in loader:
            mel, label = mel.to(device), label.to(device)
            outputs = model(mel)
            loss = criterion(outputs, label)
            loss_sum += loss.item() * label.size(0)
            preds = outputs.argmax(dim=1)
            total += label.size(0)
            correct += (preds == label).sum().item()
    avg_loss = loss_sum / total if total else 0.0
    acc = 100.0 * correct / total if total else 0.0
    return avg_loss, acc


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"

    full_dataset = GenreDataset("data/mels/")

    # Train/val split
    val_size = int(0.15 * len(full_dataset))
    train_size = len(full_dataset) - val_size
    train_ds, val_ds = random_split(
        full_dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(42),
    )

    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=16, shuffle=False)

    # Class weights for imbalance handling
    class_weights = compute_class_weights(full_dataset).to(device)

    # Infer flattened size from a sample batch
    sample_mel, _ = next(iter(train_loader))
    sample_mel = sample_mel.to(device)

    base_model = GenreCNN(num_classes=len(full_dataset.genres)).to(device)
    with torch.no_grad():
        sample_output = base_model.conv_layers(sample_mel)
        flattened_size = sample_output.view(sample_output.size(0), -1).size(1)

    model = GenreCNN(
        num_classes=len(full_dataset.genres), input_size=flattened_size
    ).to(device)

    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = optim.Adam(model.parameters(), lr=0.002, weight_decay=0.0)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )

    EPOCHS = 30
    best_val_loss = float("inf")

    for epoch in range(EPOCHS):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for mel, label in train_loader:
            mel, label = mel.to(device), label.to(device)

            # SpecAugment (keep off while pushing train accuracy)
            mel = spec_augment(mel, apply_prob=0.0)

            optimizer.zero_grad()
            outputs = model(mel)
            loss = criterion(outputs, label)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * label.size(0)
            preds = outputs.argmax(dim=1)
            total += label.size(0)
            correct += (preds == label).sum().item()

        train_loss = running_loss / total if total else 0.0
        train_acc = 100.0 * correct / total if total else 0.0

        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        scheduler.step(val_loss)

        print(
            f"Epoch {epoch+1}/{EPOCHS} "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}% "
            f"LR: {optimizer.param_groups[0]['lr']:.6f}"
        )

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), "genre_cnn.pth")

    print("Training complete. Best model saved to genre_cnn.pth")


if __name__ == "__main__":
    main()
