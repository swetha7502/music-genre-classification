import os
import librosa
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

DATASET_PATH = "data/genres_original/"
OUTPUT_PATH = "data/mels/"
MAX_LEN = 1300  # fixed number of frames for all spectrograms

genres = os.listdir(DATASET_PATH)

def save_mel_spectrogram(audio_path, mel_path):
    try:
        y, sr = librosa.load(audio_path, sr=22050)
        mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
        mel_db = librosa.power_to_db(mel, ref=np.max)

        # Pad or truncate to MAX_LEN
        if mel_db.shape[1] < MAX_LEN:
            pad_width = MAX_LEN - mel_db.shape[1]
            mel_db = np.pad(mel_db, ((0,0),(0,pad_width)), mode='constant')
        else:
            mel_db = mel_db[:, :MAX_LEN]

        np.save(mel_path, mel_db)
    except Exception as e:
        print(f"Skipping {audio_path}: {e}")

def main():
    os.makedirs(OUTPUT_PATH, exist_ok=True)
    for genre in genres:
        genre_path = os.path.join(DATASET_PATH, genre)
        out_dir = os.path.join(OUTPUT_PATH, genre)
        os.makedirs(out_dir, exist_ok=True)

        for file in tqdm(os.listdir(genre_path), desc=f"Processing {genre}"):
            audio_file = os.path.join(genre_path, file)
            mel_file = os.path.join(out_dir, file.replace(".wav", ".npy"))
            save_mel_spectrogram(audio_file, mel_file)

if __name__ == "__main__":
    main()
