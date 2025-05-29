import streamlit as st
import tempfile
import requests
import os
from pydub import AudioSegment
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch

st.title("English Accent Detector 🎙️")

video_url = st.text_input("Enter a public video URL (MP4):")

if video_url and st.button("Analyze"):
    try:
        st.info("Downloading video...")
        # Download video
        response = requests.get(video_url, stream=True)
        temp_video = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        with open(temp_video.name, 'wb') as f:
            f.write(response.content)
        st.success("Video downloaded ✅")

        # Extract audio using pydub
        st.info("Extracting audio...")
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        audio = AudioSegment.from_file(temp_video.name)
        audio.export(temp_audio.name, format="wav")
        st.success("Audio extracted ✅")

        # Load audio
        waveform, sample_rate = torchaudio.load(temp_audio.name)

        # Resample if needed
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

        # Load accent detection model
        st.info("Analyzing accent...")
        model_name = "svalabs/wav2vec2-xlsr-english-accent-classifier"
        processor = Wav2Vec2Processor.from_pretrained(model_name)
        model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

        inputs = processor(waveform.squeeze(), sampling_rate=16000, return_tensors="pt", padding=True)
        with torch.no_grad():
            logits = model(**inputs).logits
        predicted_class = torch.argmax(logits, dim=-1).item()
        confidence = torch.softmax(logits, dim=-1)[0][predicted_class].item() * 100

        labels = model.config.id2label
        accent = labels[predicted_class]

        st.success("✅ Analysis complete")
        st.markdown(f"**Accent**: {accent}")
        st.markdown(f"**Confidence**: {confidence:.2f}%")
        st.markdown(f"**Summary**: This speaker is likely using a **{accent}** English accent.")

        os.remove(temp_video.name)
        os.remove(temp_audio.name)

    except Exception as e:
        st.error(f"⚠️ Error: {e}")
