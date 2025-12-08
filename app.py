import streamlit as st
import librosa
import numpy as np
import torch
import torch.nn.functional as F
from model import GenreCNN

genres = ['blues', 'classical', 'country', 'disco', 'hiphop',
          'jazz', 'metal', 'pop', 'reggae', 'rock']

# Load model with correct input size
model = GenreCNN(num_classes=10, input_size=331776)  # Match the trained model size
model.load_state_dict(torch.load("genre_cnn.pth", map_location="cpu"))
model.eval()

def predict(audio):
    y, sr = librosa.load(audio, sr=22050, duration=30)
    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel = librosa.power_to_db(mel, ref=np.max)

    # Convert to tensor
    mel = torch.tensor(mel, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
    
    # Resize to match training dimensions
    # 331776 / 128 channels = 2592, sqrt(2592) ≈ 50.9
    # After 3 pooling layers (divide by 8), we need ~408 width to get ~51 after pooling
    # So target should be 128 x 1296 to get proper shape
    target_height = 128
    target_width = 1296
    
    mel = F.interpolate(mel, size=(target_height, target_width), 
                       mode='bilinear', align_corners=False)

    with torch.no_grad():
        out = model(mel)
    return genres[torch.argmax(out).item()]

st.title("🎵 Music Genre Classifier")
audio = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if audio:
    st.audio(audio)
    st.write("Predicting genre...")
    genre = predict(audio)
    st.success(f"🎶 Predicted Genre: **{genre}**")