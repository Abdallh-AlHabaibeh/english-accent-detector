import streamlit as st
import tempfile
import requests
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForSequenceClassification
import torch

st.title("English Accent Detector (Audio URL only)")

audio_url = st.text_input("Enter a direct audio URL (WAV or MP3):")

if audio_url and st.button("Analyze"):
    try:
        st.info("Downloading audio...")
        response = requests.get(audio_url)
        temp_audio = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
        temp_audio.write(response.content)
        temp_audio.flush()

        st.info("Loading audio...")
        waveform, sample_rate = torchaudio.load(temp_audio.name)
        # You may need to resample to 16kHz here if sample_rate != 16000

        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)

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

        st.success("Analysis complete!")
        st.markdown(f"**Accent:** {accent}")
        st.markdown(f"**Confidence:** {confidence:.2f}%")
        st.markdown(f"**Summary:** The speaker likely has a **{accent}** English accent.")

    except Exception as e:
        st.error(f"Error: {e}")
