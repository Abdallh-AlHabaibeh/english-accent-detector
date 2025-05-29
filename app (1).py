import streamlit as st
from pytube import YouTube
import tempfile
import subprocess
import os
from speechbrain.pretrained import EncoderClassifier
import torchaudio

st.title("🧠 English Accent Detector (Minimal & Real)")

video_url = st.text_input("Enter a YouTube video URL:")

def download_audio(url):
    yt = YouTube(url)
    audio_stream = yt.streams.filter(only_audio=True).first()
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    audio_stream.download(filename=temp_video.name)
    audio_path = temp_video.name.replace(".mp4", ".wav")
    subprocess.run(["ffmpeg", "-i", temp_video.name, "-ac", "1", "-ar", "16000", audio_path],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return audio_path

def classify_accent(audio_path):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-commonlanguage_ecapa",
        savedir="pretrained_models/lang-id"
    )
    prediction = classifier.classify_file(audio_path)
    predicted_accent = prediction[3][0]
    confidence = float(prediction[1][0]) * 100
    return predicted_accent, round(confidence, 2)

if st.button("Analyze"):
    if not video_url:
        st.warning("Please enter a YouTube video URL.")
    else:
        with st.spinner("Processing..."):
            try:
                audio_path = download_audio(video_url)
                accent, confidence = classify_accent(audio_path)
                st.success("✅ Accent Detected")
                st.write(f"*Accent:* {accent}")
                st.write(f"*Confidence:* {confidence}%")
                os.unlink(audio_path)
            except Exception as e:
                st.error(f"❌ Error: {e}")