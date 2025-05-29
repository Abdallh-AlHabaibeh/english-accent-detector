import streamlit as st
import tempfile
import requests
from moviepy.editor import VideoFileClip
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch
import os

st.title("English Accent Detection Tool 🎤")

video_url = st.text_input("Enter a public video URL (MP4 or Loom):")

if video_url and st.button("Analyze Accent"):
    st.info("Downloading video...")
    try:
        # Download video
        response = requests.get(video_url, stream=True)
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_video.name, 'wb') as f:
            f.write(response.content)

        st.success("Video downloaded!")

        # Extract audio
        st.info("Extracting audio...")
        video = VideoFileClip(temp_video.name)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        video.audio.write_audiofile(temp_audio.name, codec='pcm_s16le')
        st.success("Audio extracted!")

        # Load audio
        waveform, sample_rate = torchaudio.load(temp_audio.name)

        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000

        # Load model
        st.info("Running accent classification...")
        model_name = "svalabs/wav2vec2-xlsr-english-accent-classifier"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

        inputs = processor(waveform.squeeze(), sampling_rate=sample_rate, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1)[0][predicted_class].item() * 100

        # Label mapping
        labels = model.config.id2label
        accent_label = labels[predicted_class]

        st.success("Analysis complete!")
        st.subheader("📝 Results")
        st.write(f"**Detected Accent**: {accent_label}")
        st.write(f"**Confidence Score**: {confidence:.2f}%")
        st.write(f"**Summary**: This speaker most likely has a(n) {accent_label} accent based on English speech patterns.")

        # Cleanup
        os.remove(temp_video.name)
        os.remove(temp_audio.name)

    except Exception as e:
        st.error(f"Something went wrong: {e}")
