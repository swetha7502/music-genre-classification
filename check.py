from pydub import AudioSegment

file_path = "data/genres_original/jazz/jazz.00054.wav"
fixed_path = "data/genres_original/jazz/jazz.00054_fixed.wav"

try:
    audio = AudioSegment.from_file(file_path)
    audio.export(fixed_path, format="wav")
    print("File fixed!")
except Exception as e:
    print("Cannot fix:", e)
