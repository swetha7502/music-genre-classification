# Music Genre Classification

Deep learning pipeline that classifies 30-second audio clips into one of 10 genres (blues, classical, country, disco, hiphop, jazz, metal, pop, reggae, rock) using mel-spectrograms and a small CNN. Includes preprocessing, training, a prediction helper, and a Streamlit demo.

## Project Layout
- `app.py` – Streamlit UI for uploading an audio file and displaying the predicted genre (loads `genre_cnn.pth` with inferred input size).
- `predict.py` – Reusable `predict_genre()` helper for offline inference.
- `preprocess.py` – Converts raw audio into mel-spectrogram `.npy` files.
- `train.py` – Trains the CNN on the processed mel data (train/val split, class weights, LR scheduler) and saves `genre_cnn.pth`.
- `dataset.py` / `model.py` – Dataset wrapper (per-sample normalization) and CNN architecture (adaptive pooling, 256-dim FC).
- `data/` – Expected location of the dataset (`genres_original/`) and generated mel features (`mels/`).
- `genre_cnn.pth` – Saved weights for inference (generated after training).
- `ARCHITECTURE.md` – Mermaid diagrams of data flow, model, and inference paths.

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
Train the CNN on the processed mel data (with train/val split, class weights, LR scheduler):
```bash
python train.py
```
This infers the FC input size from a sample batch, trains for 30 epochs (best model saved by val loss) and writes `genre_cnn.pth`.

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
The app normalizes mel-spectrograms to match training and infers the FC size automatically.

### UI Preview

Home screen (custom dark theme with coral accent):

![Streamlit home 1](assets/Screenshot%202025-12-08%20at%202.28.00%E2%80%AFAM.png)

![Streamlit home 2](assets/Screenshot%202025-12-08%20at%202.28.57%E2%80%AFAM.png)

![Streamlit home 3](assets/Screenshot%202025-12-08%20at%202.29.32%E2%80%AFAM.png)

![Streamlit home 4](assets/Screenshot%202025-12-08%20at%202.30.01%E2%80%AFAM.png)

## Evaluation
Run on held-out test split:
```bash
python evaluate.py
```
Latest run: **Test accuracy 80.40%**, Macro F1 0.802 (confusion matrix saved to `assets/confusion_matrix.png`).

## Notes
- `check.py` contains a small utility to re-export problematic audio files with pydub.
- If you retrain with different input shapes, the app/predict path will infer the flattened size; you can still pass `input_size` manually if needed (see `train.py` for how it is derived).
