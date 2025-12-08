import streamlit as st
import librosa
import librosa.display
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

from model import GenreCNN

if "history" not in st.session_state:
    st.session_state.history = []

GENRE_FACTS = {
    "rock": "Rock often features electric guitars, strong drums, and high energy.",
    "classical": "Classical pieces can span from Baroque to Romantic eras over centuries.",
    "jazz": "Jazz is all about improvisation and swing rhythms.",
    "hiphop": "Hip hop blends rhythm, rhyme, and spoken word over beats.",
    "reggae": "Reggae grew out of Jamaica, known for off-beat rhythms and chill vibes."
}

# -----------------------------
# PAGE SETUP & BASIC STYLING
# -----------------------------
st.set_page_config(
    page_title="Music Genre Classifier",
    page_icon="🎵",
    layout="wide",
)

# Minimal custom CSS
st.markdown(
    """
    <style>
    .stApp {
        background: radial-gradient(circle at top, #262626 0, #050505 45%, #000000 100%);
        color: #ffffff;
    }
    .card {
        background-color: #151515;
        border-radius: 16px;
        padding: 20px 24px;
        box-shadow: 0 10px 25px rgba(0,0,0,0.4);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------
# MODEL LOADING
# -----------------------------
genres = [
    "blues", "classical", "country", "disco", "hiphop",
    "jazz", "metal", "pop", "reggae", "rock"
]

@st.cache_resource
def load_model():
    model = GenreCNN(num_classes=10, input_size=331776)
    checkpoint = torch.load("genre_cnn.pth", map_location="cpu")
    model.load_state_dict(checkpoint)
    model.eval()
    return model

model = load_model()

# -----------------------------
# PREDICTION PIPELINE
# -----------------------------
def predict(audio_file):
    """
    audio_file is a Streamlit UploadedFile.
    Returns: predicted_genre, probs, y, sr, mel_db
    """
    y, sr = librosa.load(audio_file, sr=22050, duration=30)

    mel = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    mel_db = librosa.power_to_db(mel, ref=np.max)

    mel_tensor = torch.tensor(mel_db, dtype=torch.float32).unsqueeze(0).unsqueeze(0)

    target_height = 128
    target_width = 1296
    mel_tensor = F.interpolate(
        mel_tensor,
        size=(target_height, target_width),
        mode="bilinear",
        align_corners=False,
    )

    with torch.no_grad():
        out = model(mel_tensor)
        probs = torch.softmax(out, dim=1)[0].cpu().numpy()
        pred_idx = int(np.argmax(probs))

    predicted_genre = genres[pred_idx]
    return predicted_genre, probs, y, sr, mel_db

# -----------------------------
# HEADER
# -----------------------------
st.markdown(
    """
    <h1 style='text-align:center; font-size:44px; margin-bottom:0'>
        🎵 Music Genre Classifier
    </h1>
    <p style='text-align:center; color:#cccccc; font-size:16px; margin-top:4px;'>
        Upload a track and see which genre our CNN thinks it belongs to.
    </p>
    """,
    unsafe_allow_html=True,
)
st.markdown("<br>", unsafe_allow_html=True)

# -----------------------------
# MAIN LAYOUT
# -----------------------------
col_left, col_right = st.columns([1.2, 1])

with col_left:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Upload an audio file")
    audio = st.file_uploader(
        "Drag & drop or browse (WAV / MP3)",
        type=["wav", "mp3"],
    )

    if audio is not None:
        st.audio(audio, format="audio/mp3")
    st.markdown("</div>", unsafe_allow_html=True)

with col_right:
    st.markdown("<div class='card'>", unsafe_allow_html=True)
    st.subheader("Prediction")

    if audio is None:
        st.info("Upload an audio file to get a prediction.")
        predicted_genre = None
        probs = None
        y = sr = mel_db = None
    else:
        with st.spinner("Analyzing audio..."):
            predicted_genre, probs, y, sr, mel_db = predict(audio)

        # main prediction pill
        st.success(f"🎶 Predicted Genre: **{predicted_genre}**")

        # top-3 genres as chips
        top_idxs = np.argsort(probs)[::-1][:3]
        chip_html = "<div style='margin-top:8px;'>"
        for i in top_idxs:
            g = genres[i]
            p = probs[i]
            chip_html += (
                f"<span style='display:inline-block; padding:4px 10px; "
                f"border-radius:999px; background:#222; color:#ddd; "
                f"margin-right:6px; font-size:12px;'>"
                f"{g} &nbsp; {p*100:.1f}%</span>"
            )
        chip_html += "</div>"
        st.markdown(chip_html, unsafe_allow_html=True)

        # bar chart
        prob_dict = {g: float(p) for g, p in zip(genres, probs)}
        st.write("Model confidence by genre:")
        st.bar_chart(prob_dict)

        fact = GENRE_FACTS.get(predicted_genre)
        if fact:
            st.caption(f"💡 Fun fact about **{predicted_genre}**: {fact}")


    st.markdown("</div>", unsafe_allow_html=True)

# -----------------------------
# EXTRA VISUALS (ONLY IF AUDIO)
# -----------------------------
if audio is not None:
    st.markdown("<br>", unsafe_allow_html=True)
    vcol1, vcol2 = st.columns(2)

    with vcol1:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Waveform**")
        fig, ax = plt.subplots(figsize=(8, 2))
        librosa.display.waveshow(y, sr=sr, ax=ax)
        ax.set_xlabel("Time (s)")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    with vcol2:
        st.markdown("<div class='card'>", unsafe_allow_html=True)
        st.markdown("**Mel-Spectrogram (what the model sees)**")
        fig, ax = plt.subplots(figsize=(8, 3))
        img = librosa.display.specshow(
            mel_db, sr=sr, x_axis="time", y_axis="mel", ax=ax
        )
        fig.colorbar(img, ax=ax, format="%+2.0f dB")
        st.pyplot(fig)
        st.markdown("</div>", unsafe_allow_html=True)

    st.session_state.history.append({"filename": audio.name, "genre": predicted_genre})

st.sidebar.title("Prediction History")
if st.session_state.history:
    for item in reversed(st.session_state.history[-5:]):
        st.sidebar.write(f"• **{item['genre']}** – {item['filename']}")
else:
    st.sidebar.write("No predictions yet.")


# -----------------------------
# FOOTER
# -----------------------------
st.markdown(
    "<hr><p style='text-align:center; color:#777;'>Built by Siri & Swetha • GTZAN Dataset • CNN + Mel-Spectrograms</p>",
    unsafe_allow_html=True,
)
