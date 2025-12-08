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


import torch
import librosa
import numpy as np
import torch.nn.functional as F
from model import GenreCNN

genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load checkpoint
checkpoint = torch.load("genre_cnn.pth", map_location='cpu')

# Create model with saved configuration
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model = GenreCNN(num_classes=checkpoint['num_classes'], 
                     input_size=checkpoint['input_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model = GenreCNN(num_classes=10, input_size=331776)
    model.load_state_dict(checkpoint)

model.eval()

def predict_genre(audio_path):
    # Load audio
    y, sr = librosa.load(audio_path, sr=22050, duration=30)  # Load 30 seconds
    
    # Generate mel-spectrogram
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)
    
    # Convert to tensor
    mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Resize to match training dimensions
    # 331776 = 128 * 648 * 4 (from 3 pooling layers: 128 channels, and spatial dims)
    # Let's calculate what the original input should be
    # After 3 MaxPool2d(2): spatial dims are divided by 8
    # So original should be around 128 x 1296 to get 128 x 162 after pooling
    
    target_height = 128
    target_width = 1296  # This gives ~162 after 3 pooling layers
    
    mel_tensor = F.interpolate(mel_tensor, size=(target_height, target_width), 
                               mode='bilinear', align_corners=False)
    
    # Predict
    with torch.no_grad():
        out = model(mel_tensor)
        genre_idx = torch.argmax(out, dim=1).item()
    
    return genres[genre_idx]

print(predict_genre("data\genres_original\metal\metal.00000.wav"))