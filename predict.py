# import torch
# import librosa
# import numpy as np
# from model import GenreCNN

# genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
#           'jazz', 'metal', 'pop', 'reggae', 'rock']

# model = GenreCNN(num_classes=10)
# model.load_state_dict(torch.load("genre_cnn.pth"))
# model.eval()

# def predict_genre(audio_path):
#     y, sr = librosa.load(audio_path, sr=22050)
#     mel = librosa.feature.melspectrogram(y=y, sr=sr)
#     mel_db = librosa.power_to_db(mel, ref=np.max)

#     mel_db = np.expand_dims(mel_db, axis=0)
#     mel_db = np.expand_dims(mel_db, axis=0)
#     mel_db = torch.tensor(mel_db, dtype=torch.float32)

#     with torch.no_grad():
#         out = model(mel_db)
#         genre_idx = torch.argmax(out, dim=1).item()

#     return genres[genre_idx]

# print(predict_genre("example.wav"))

import os
import torch
import librosa
import numpy as np
import torch.nn.functional as F
from model import GenreCNN

genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# ---- Load trained model ----
checkpoint = torch.load("genre_cnn.pth", map_location="cpu")

if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    num_classes = checkpoint.get("num_classes", 10)
    input_size = checkpoint.get("input_size", 331776)
    model = GenreCNN(num_classes=num_classes, input_size=input_size)
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    # Fallback: assume we saved just the state_dict
    model = GenreCNN(num_classes=10, input_size=331776)
    model.load_state_dict(checkpoint)

model.eval()


def predict_genre(audio_path: str) -> str:
    """
    Predict the genre of a given audio file.
    """
    # Load up to 30 seconds of audio
    y, sr = librosa.load(audio_path, sr=22050, duration=30)

    # Create mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    # Convert to 4D tensor: (batch, channel, height, width)
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    # Resize to match training dimensions
    target_height = 128
    target_width = 1296  # chosen to match what the CNN expects after pooling

    mel_tensor = F.interpolate(
        mel_tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    # Run through model
    with torch.no_grad():
        out = model(mel_tensor)
        genre_idx = torch.argmax(out, dim=1).item()

    return genres[genre_idx]


if __name__ == "__main__":
    # Use POSIX-style / or os.path.join (works on Mac/Win/Linux)
    test_path = os.path.join("data", "genres_original", "metal", "metal.00000.wav")
    print("Testing on:", test_path)
    print("Predicted genre:", predict_genre(test_path))
