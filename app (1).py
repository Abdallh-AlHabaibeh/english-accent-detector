import streamlit as st
import yt_dlp
import os
import subprocess
from speechbrain.pretrained import EncoderClassifier
import torch
import numpy as np

# Download video from URL and extract audio path
def download_video_extract_audio(video_url, output_dir="downloads"):
    os.makedirs(output_dir, exist_ok=True)
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': f'{output_dir}/video.%(ext)s',
        'quiet': True,
        'no_warnings': True,
    }
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info_dict = ydl.extract_info(video_url, download=True)
        ext = info_dict.get('ext', 'mp4')
        video_path = f"{output_dir}/video.{ext}"
        
    # Extract audio with ffmpeg to WAV 16kHz mono
    audio_path = f"{output_dir}/audio.wav"
    cmd = [
        "ffmpeg", "-y", "-i", video_path,
        "-ac", "1", "-ar", "16000",
        "-vn", audio_path
    ]
    subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return audio_path

# Mock classifier: accepts speechbrain embedding, returns accent + confidence
def classify_accent(embedding):
    # For demo, randomly assign accents based on embedding norm (fake logic)
    norm = torch.norm(embedding).item()
    accents = ['American', 'British', 'Australian', 'Indian']
    idx = int(norm * 10) % len(accents)
    accent = accents[idx]
    confidence = min(100, (norm * 50) % 100)
    summary = f"The speaker most likely has a {accent} English accent with confidence {confidence:.1f}%."
    return accent, confidence, summary

@st.cache_data(show_spinner=False)
def get_speaker_embedding(audio_path):
    classifier = EncoderClassifier.from_hparams(source="speechbrain/spkrec-ecapa-voxceleb", run_opts={"device":"cpu"})
    signal, fs = classifier.load_audio(audio_path)
    embedding = classifier.encode_batch(signal)
    return embedding.squeeze()

def main():
    st.title("English Accent Detection from Video URL")
    st.write("Enter a public video URL (e.g., Loom, direct MP4).")
    
    video_url = st.text_input("Video URL:")
    if st.button("Analyze Accent"):
        if not video_url.strip():
            st.error("Please enter a valid video URL.")
            return
        
        with st.spinner("Downloading video and extracting audio..."):
            try:
                audio_path = download_video_extract_audio(video_url)
            except Exception as e:
                st.error(f"Failed to download or extract audio: {e}")
                return
        
        with st.spinner("Analyzing accent..."):
            try:
                embedding = get_speaker_embedding(audio_path)
                accent, confidence, summary = classify_accent(embedding)
            except Exception as e:
                st.error(f"Accent analysis failed: {e}")
                return
        
        st.success("Analysis complete!")
        st.write(f"**Accent Classification:** {accent}")
        st.write(f"**Confidence:** {confidence:.1f}%")
        st.write(f"**Summary:** {summary}")

if __name__ == "__main__":
    main()
