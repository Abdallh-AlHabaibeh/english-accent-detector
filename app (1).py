import streamlit as st
import tempfile
import os
import subprocess
from pytube import YouTube
import whisper
import torchaudio
from speechbrain.pretrained import EncoderClassifier

# Download video and extract audio
def download_and_extract_audio(video_url):
    try:
        yt = YouTube(video_url)
        stream = yt.streams.filter(only_audio=True).first()
        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        stream.download(filename=temp_video_file.name)

        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        subprocess.call([
            'ffmpeg', '-y',
            '-i', temp_video_file.name,
            '-ar', '16000',
            '-ac', '1',
            temp_audio_file.name
        ])

        os.unlink(temp_video_file.name)
        return temp_audio_file.name
    except Exception as e:
        st.error(f"Error processing video: {e}")
        return None

# Transcribe audio using Whisper
def transcribe_audio(audio_path):
    model = whisper.load_model("base")
    result = model.transcribe(audio_path)
    return result["text"]

# Classify accent using SpeechBrain
def classify_accent(audio_path):
    classifier = EncoderClassifier.from_hparams(
        source="speechbrain/lang-id-commonlanguage_ecapa",
        savedir="pretrained_models/lang-id-commonlanguage_ecapa"
    )
    prediction = classifier.classify_file(audio_path)
    accent = prediction[3][0]
    confidence = prediction[1][0].item() * 100
    return accent, confidence

# Streamlit UI
st.set_page_config(page_title="English Accent Detector", page_icon="🎧")
st.title("🎙️ English Accent Detection Tool")

video_url = st.text_input("Enter a YouTube video URL:")

if st.button("Analyze"):
    if video_url:
        with st.spinner("Processing..."):
            audio_path = download_and_extract_audio(video_url)
            if audio_path:
                transcript = transcribe_audio(audio_path)
                accent, confidence = classify_accent(audio_path)
                st.success("Analysis complete!")
                st.markdown(f"**Transcript:** {transcript}")
                st.markdown(f"**Detected Accent:** `{accent}`")
                st.markdown(f"**Confidence Score:** `{confidence:.2f}%`")
                os.unlink(audio_path)
    else:
        st.warning("Please enter a YouTube URL.")
