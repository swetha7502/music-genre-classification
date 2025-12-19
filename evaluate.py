import os

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split

from dataset import GenreDataset
from model import GenreCNN


def compute_flattened_size(sample_batch, device):
    """Infer the flattened feature size after conv/pool stack."""
    probe_model = GenreCNN(num_classes=10).to(device)
    with torch.no_grad():
        out = probe_model.conv_layers(sample_batch.to(device))
        flat_size = out.view(out.size(0), -1).size(1)
    return flat_size


def load_trained_model(flattened_size, device):
    """Load the trained CNN with the correct FC input size."""
    state = torch.load("genre_cnn.pth", map_location=device)
    model = GenreCNN(num_classes=10, input_size=flattened_size).to(device)
    model.load_state_dict(state)
    model.eval()
    return model


def evaluate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Dataset and split
    full_ds = GenreDataset("data/mels/")
    test_size = int(0.2 * len(full_ds))
    train_size = len(full_ds) - test_size
    train_ds, test_ds = random_split(
        full_ds, [train_size, test_size], generator=torch.Generator().manual_seed(42)
    )

    # Infer flattened size from a sample batch
    sample_batch, _ = next(iter(DataLoader(train_ds, batch_size=1, shuffle=False)))
    flat_size = compute_flattened_size(sample_batch, device)

    model = load_trained_model(flattened_size=flat_size, device=device)

    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False)

    correct = 0
    total = 0
    class_names = full_ds.genres
    num_classes = len(class_names)
    confusion = torch.zeros((num_classes, num_classes), dtype=torch.int64)

    with torch.no_grad():
        for mels, labels in test_loader:
            mels = mels.to(device)
            labels = labels.to(device)

            outputs = model(mels)
            preds = outputs.argmax(dim=1)

            correct += (preds == labels).sum().item()
            total += labels.size(0)

            for t, p in zip(labels.view(-1), preds.view(-1)):
                confusion[t.long(), p.long()] += 1

    acc = 100.0 * correct / total if total else 0.0
    print(f"Test accuracy: {acc:.2f}% ({correct}/{total})")
    print("Confusion matrix (rows=true, cols=pred):")
    conf_np = confusion.cpu().numpy()
    print(conf_np)

    # Derived metrics
    true_pos = np.diag(conf_np)
    support = conf_np.sum(axis=1)
    pred_total = conf_np.sum(axis=0)
    precision = np.divide(
        true_pos,
        pred_total,
        out=np.zeros_like(true_pos, dtype=float),
        where=pred_total != 0,
    )
    recall = np.divide(
        true_pos,
        support,
        out=np.zeros_like(true_pos, dtype=float),
        where=support != 0,
    )
    f1 = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(true_pos, dtype=float),
        where=(precision + recall) != 0,
    )
    macro_precision = precision.mean() if len(precision) else 0.0
    macro_recall = recall.mean() if len(recall) else 0.0
    macro_f1 = f1.mean() if len(f1) else 0.0

    print("\nPer-class metrics:")
    for idx, name in enumerate(class_names):
        print(
            f"  {name:<12} P: {precision[idx]:.3f} | R: {recall[idx]:.3f} | F1: {f1[idx]:.3f} | Support: {support[idx]}"
        )
    print(
        f"\nMacro Precision: {macro_precision:.3f}, Macro Recall: {macro_recall:.3f}, Macro F1: {macro_f1:.3f}"
    )

    # Plot and save confusion matrix
    os.makedirs("assets", exist_ok=True)
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(confusion.cpu().numpy(), interpolation="nearest", cmap="Blues")
    ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    ax.set(
        xticks=range(num_classes),
        yticks=range(num_classes),
        xticklabels=class_names,
        yticklabels=class_names,
        ylabel="True label",
        xlabel="Predicted label",
        title=f"Confusion Matrix (Acc: {acc:.2f}%)",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    for i in range(num_classes):
        for j in range(num_classes):
            ax.text(
                j,
                i,
                f"{confusion[i, j].item()}",
                ha="center",
                va="center",
                color="black" if confusion[i, j] < confusion.max() / 2 else "white",
                fontsize=8,
            )
    fig.tight_layout()
    out_path = os.path.join("assets", "confusion_matrix.png")
    plt.savefig(out_path, dpi=200)
    plt.close(fig)
    print(f"Saved confusion matrix to {out_path}")


if __name__ == "__main__":
    evaluate()
