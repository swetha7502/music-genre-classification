# Music Genre Classification

Deep learning pipeline that classifies 30-second audio clips into one of 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) using mel-spectrograms and a small CNN. Includes preprocessing, training, a prediction helper, and a Streamlit demo.

## Project Layout
- `app.py` – Streamlit UI for uploading an audio file and displaying the predicted genre.
- `predict.py` – Reusable `predict_genre()` helper for offline inference.
- `preprocess.py` – Converts raw audio into fixed-length mel-spectrogram `.npy` files.
- `train.py` – Trains the CNN on the processed mel data and saves `genre_cnn.pth`.
- `dataset.py` / `model.py` – Dataset wrapper and CNN architecture.
- `data/` – Expected location of the dataset (`genres_original/`) and generated mel features (`mels/`).
- `genre_cnn.pth` – Saved weights for inference (generated after training).

## Requirements
- Python 3.9+ recommended
- Install dependencies:
  ```bash
  pip install torch torchvision torchaudio librosa numpy streamlit tqdm matplotlib pydub
  ```

## Data
- Uses the GTZAN-like layout: `data/genres_original/<genre>/<file>.wav`.
- The dataset is **not** included. Download GTZAN (or your own 10-genre dataset) and place it under `data/genres_original/`.

## Preprocessing
Generate mel-spectrogram tensors for training:
```bash
python preprocess.py
```
Outputs are written to `data/mels/<genre>/<file>.npy`, padded/truncated to a fixed width (`MAX_LEN`).

## Training
Train the CNN on the processed mel data:
```bash
python train.py
```
This recalculates the fully connected input size from a sample batch, trains for 15 epochs, and writes `genre_cnn.pth` for later inference.

## Inference (CLI)
Use the helper to predict a single file:
```bash
python - <<'PY'
from predict import predict_genre
print(predict_genre("data/genres_original/metal/metal.00000.wav"))
PY
```
Ensure `genre_cnn.pth` is present in the project root.

## Streamlit App
Launch the web demo to upload `.wav`/`.mp3` files and view the predicted genre:
```bash
streamlit run app.py
```
The app resizes mel-spectrograms to the training dimensions before inference.

## Notes
- `check.py` contains a small utility to re-export problematic audio files with pydub.
- If you retrain with different input shapes, pass the new `input_size` when instantiating `GenreCNN` for inference (see `train.py` for how it is derived).
