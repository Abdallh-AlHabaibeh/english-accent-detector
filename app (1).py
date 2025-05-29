import streamlit as st
import tempfile
import os
import subprocess
from pytube import YouTube
import speech_recognition as sr
from pytube import YouTube
from pytube.exceptions import VideoUnavailable
import requests

def download_and_extract_audio(video_url):
    try:
        if not video_url.endswith(".mp4"):
            st.error("Only direct MP4 links are supported in this version.")
            return None

        video_response = requests.get(video_url, stream=True)
        if video_response.status_code != 200:
            st.error("Failed to download video.")
            return None

        temp_video_file = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_video_file.name, 'wb') as f:
            for chunk in video_response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)

        temp_audio_file = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        subprocess.call([
            'ffmpeg',
            '-i', temp_video_file.name,
            '-ar', '16000',
            '-ac', '1',
            temp_audio_file.name
        ])
        os.unlink(temp_video_file.name)
        return temp_audio_file.name

    except Exception as e:
        st.error(f"Error downloading or processing video: {e}")
        return None



def transcribe_audio(audio_path):
    recognizer = sr.Recognizer()
    with sr.AudioFile(audio_path) as source:
        audio = recognizer.record(source)
    try:
        text = recognizer.recognize_google(audio)
        return text
    except sr.UnknownValueError:
        return ""
    except sr.RequestError:
        return ""

def simple_accent_classifier(text):
    # Basic heuristic for demo
    british_words = ['colour', 'favour', 'centre', 'theatre']
    american_words = ['color', 'favor', 'center', 'theater']
    
    british_score = sum(word in text.lower() for word in british_words)
    american_score = sum(word in text.lower() for word in american_words)
    
    if british_score > american_score:
        return "British", british_score / max(len(british_words), 1) * 100
    elif american_score > british_score:
        return "American", american_score / max(len(american_words), 1) * 100
    else:
        return "Unknown", 0

st.title("🎙️ English Accent Detection Demo")

video_url = st.text_input("Enter a public video URL (YouTube or direct MP4 link):")

if st.button("Analyze"):
    if video_url:
        with st.spinner("Processing..."):
            audio_path = download_and_extract_audio(video_url)
            if audio_path:
                transcript = transcribe_audio(audio_path)
                if transcript:
                    accent, confidence = simple_accent_classifier(transcript)
                    st.write(f"**Transcript:** {transcript}")
                    st.write(f"**Detected Accent:** {accent}")
                    st.write(f"**Confidence Score:** {confidence:.2f}%")
                else:
                    st.error("Could not transcribe audio.")
                os.unlink(audio_path)
            else:
                st.error("Failed to extract audio.")
    else:
        st.warning("Please enter a valid video URL.")
