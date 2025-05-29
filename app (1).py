import streamlit as st
import requests
import time

st.title("English Accent Detection")

API_KEY = "e5cc70299daf4eed810ff8e6f7684e32"  

HEADERS = {
    "authorization": API_KEY,
    "content-type": "application/json"
}

def request_transcription(video_url):
    json_data = {
        "audio_url": video_url,
        "language_detection": True
    }
    response = requests.post(
        'https://api.assemblyai.com/v2/transcript',
        json=json_data,
        headers=HEADERS
    )
    response.raise_for_status()
    return response.json()['id']

def get_transcription_result(transcript_id):
    while True:
        response = requests.get(f'https://api.assemblyai.com/v2/transcript/{transcript_id}', headers=HEADERS)
        response.raise_for_status()
        result = response.json()
        if result['status'] == 'completed':
            return result
        elif result['status'] == 'error':
            raise Exception("Transcription failed: " + result.get('error', 'Unknown error'))
        else:
            time.sleep(3)

video_url = st.text_input("Enter direct video URL (public MP4 URL):")

if video_url:
    try:
        with st.spinner("Requesting transcription..."):
            transcript_id = request_transcription(video_url)

        with st.spinner("Waiting for transcription..."):
            result = get_transcription_result(transcript_id)

        st.subheader("Transcription:")
        st.write(result.get("text", ""))

        st.subheader("Detected Language Info:")
        language = result.get('language', {})
        if language:
            st.write(f"Language: {language.get('language')}")
            st.write(f"Confidence: {language.get('confidence'):.2f}")
            st.write(f"Dialect: {language.get('dialect')}")
        else:
            st.write("Language detection info not available.")

    except Exception as e:
        st.error(f"Error: {e}")
