# Use docker compose so use it
# import streamlit as st
# import requests
#
# st.title("🎙 AI Audio Captioning Demo")
#
# uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])
#
# if uploaded_file is not None:
#     st.audio(uploaded_file)
#
#     # Gửi file lên FastAPI backend
#     files = {"file": (uploaded_file.name, uploaded_file, "audio/wav")}
#     response = requests.post("http://backend:8000/upload-audio/", files=files)
#
#     if response.status_code == 200:
#         data = response.json()
#         if "caption" in data:
#             st.success("Generated Caption:")
#             st.write(data["caption"])
#         else:
#             st.error(f"Error: {data.get('error', 'Unknown error')}")
#     else:
#         st.error(f"API Error: {response.status_code}")


# import streamlit as st
# import requests
# from config import BACKEND_URL
#
# st.title("🎙 AI Audio Captioning Demo")
#
# uploaded_file = st.file_uploader("Upload an audio file (.wav)", type=["wav"])
#
# if uploaded_file is not None:
#     st.audio(uploaded_file)
#
#     files = {"file": (uploaded_file.name, uploaded_file, "audio/wav")}
#     try:
#         response = requests.post(f"{BACKEND_URL}/upload-audio/", files=files)
#         if response.status_code == 200:
#             data = response.json()
#             if "caption" in data:
#                 st.success("Generated Caption:")
#                 st.write(data["caption"])
#             else:
#                 st.error(f"Error: {data.get('error', 'Unknown error')}")
#         else:
#             st.error(f"API Error: {response.status_code}")
#     except Exception as e:
#         st.error(f"Connection failed: {str(e)}")


# import streamlit as st
# import requests
# import threading
# import time
#
# BACKEND_URL = "http://localhost:8000"
#
# st.title("🎙 AI Audio Captioning + Realtime Summarization")
#
# # Section Upload file
# st.subheader("📤 Upload file audio")
# uploaded_file = st.file_uploader("Chọn file .wav", type=["wav"])
# if uploaded_file is not None:
#     st.audio(uploaded_file)
#     files = {"file": (uploaded_file.name, uploaded_file, "audio/wav")}
#     response = requests.post(f"{BACKEND_URL}/upload-audio/", files=files)
#     if response.status_code == 200:
#         st.success("Generated Caption:")
#         st.write(response.json()["caption"])
#     else:
#         st.error("Error upload!")
#
# # Section realtime stream polling
# st.subheader("🎙 Realtime caption")
#
# placeholder = st.empty()
#
# def poll_caption():
#     while True:
#         try:
#             res = requests.get(f"{BACKEND_URL}/latest-caption/")
#             caption = res.json().get("caption", "")
#             if caption:
#                 placeholder.write(f"📝 {caption}")
#         except Exception as e:
#             print(e)
#         time.sleep(1)
#
# # Bắt đầu polling ngay khi frontend khởi động
# threading.Thread(target=poll_caption, daemon=True).start()
#
# # Nút Start Microphone Streaming
# if st.button("🎙 Start Microphone Streaming"):
#     response = requests.post(f"{BACKEND_URL}/start-mic-streaming/")
#     if response.status_code == 200:
#         st.success("Microphone streaming has started.")
#     else:
#         st.error("Failed to start streaming.")
#
# # Nút tóm tắt
# if st.button("📄 Sinh tóm tắt nhanh"):
#     res = requests.get(f"{BACKEND_URL}/summarize/")
#     if res.status_code == 200:
#         st.success(res.json()["summary"])
#     else:
#         st.error("Error generating summary")



# import streamlit as st
# import requests
# import threading
# import time
# import sounddevice as sd
# import numpy as np
#
# BACKEND_URL = "http://localhost:8000"
#
# st.title("🎙 AI Audio Captioning + Realtime Summarization")
#
# # --- Upload file section ---
# st.subheader("📤 Upload file audio")
# uploaded_file = st.file_uploader("Chọn file .wav", type=["wav"])
# if uploaded_file is not None:
#     st.audio(uploaded_file)
#     files = {"file": (uploaded_file.name, uploaded_file, "audio/wav")}
#     response = requests.post(f"{BACKEND_URL}/upload-audio/", files=files)
#     if response.status_code == 200:
#         st.success("Generated Caption:")
#         st.write(response.json()["caption"])
#     else:
#         st.error("Error upload!")
#
# # --- Realtime caption polling ---
# st.subheader("🎙 Realtime caption")
# placeholder = st.empty()
#
# def poll_caption():
#     while True:
#         try:
#             res = requests.get(f"{BACKEND_URL}/latest-caption/")
#             caption = res.json().get("caption", "")
#             if caption:
#                 placeholder.write(f"📝 {caption}")
#         except:
#             pass
#         time.sleep(1)
#
# threading.Thread(target=poll_caption, daemon=True).start()
#
# def visualize_audio(indata, frames, time, status):
#     volume_norm = np.linalg.norm(indata) * 10
#     st.session_state.audio_level = volume_norm
#
# if st.button("🎙 Start Microphone Streaming"):
#     res = requests.post(f"{BACKEND_URL}/start-mic-streaming/")
#     if res.status_code == 200:
#         st.session_state.mic_active = True
#         st.success("Microphone streaming has started.")
#
# if st.button("🛑 Stop Microphone Streaming"):
#     res = requests.post(f"{BACKEND_URL}/stop-mic-streaming/")
#     if res.status_code == 200:
#         st.session_state.mic_active = False
#         st.success("Microphone streaming has stopped.")
#
# # --- Summarization ---
# st.subheader("📄 Quick Summary")
# if st.button("📄 Sinh tóm tắt nhanh"):
#     res = requests.get(f"{BACKEND_URL}/summarize/")
#     if res.status_code == 200:
#         st.success(res.json()["summary"])




import streamlit as st
import requests
import threading
import time
import numpy as np
import pandas as pd

BACKEND_URL = "http://localhost:8000"

st.title("🎙 Audio Captioning And Realtime Summarization")

# Section Upload file
st.subheader("📤 Upload file audio")
uploaded_file = st.file_uploader("Chọn file .wav", type=["wav"])
if uploaded_file is not None:
    st.audio(uploaded_file)
    files = {"file": (uploaded_file.name, uploaded_file, "audio/wav")}
    response = requests.post(f"{BACKEND_URL}/upload-audio/", files=files)
    if response.status_code == 200:
        st.success("Generated Caption:")
        st.write(response.json()["caption"])
    else:
        st.error("Error upload!")

# Section realtime stream polling
st.subheader("🎙 Realtime caption")

caption_placeholder = st.empty()
mic_level_placeholder = st.empty()

# Realtime polling caption
def poll_caption():
    while True:
        try:
            res = requests.get(f"{BACKEND_URL}/latest-caption/")
            caption = res.json().get("caption", "")
            caption_placeholder.write(f"📝 {caption}")
        except:
            pass
        time.sleep(1)

# Realtime polling mic level
def poll_mic_level():
    while True:
        try:
            res = requests.get(f"{BACKEND_URL}/mic-level/")
            level = res.json().get("level", 0)
            bar = "🔊" * (level // 2)
            mic_level_placeholder.markdown(f"**Mic Level:** {bar}")
        except:
            pass
        time.sleep(0.5)

waveform_placeholder = st.empty()

def poll_waveform():
    while True:
        try:
            res = requests.get(f"{BACKEND_URL}/waveform/")
            data = res.json().get("waveform", [])
            if data:
                df = pd.DataFrame(data, columns=["amplitude"])
                waveform_placeholder.line_chart(df)
        except:
            pass
        time.sleep(0.5)

threading.Thread(target=poll_waveform, daemon=True).start()

# Khởi động 2 luồng polling song song
threading.Thread(target=poll_caption, daemon=True).start()
threading.Thread(target=poll_mic_level, daemon=True).start()

# Start/Stop Microphone Streaming
col1, col2 = st.columns(2)

with col1:
    if st.button("🎙 Start Microphone Streaming"):
        response = requests.post(f"{BACKEND_URL}/start-mic-streaming/")
        if response.status_code == 200:
            st.success("Microphone streaming has started.")
        else:
            st.error("Failed to start streaming.")

with col2:
    if st.button("🛑 Stop Microphone Streaming"):
        response = requests.post(f"{BACKEND_URL}/stop-mic-streaming/")
        if response.status_code == 200:
            st.success("Microphone streaming stopped.")
        else:
            st.error("Failed to stop streaming.")

# Quick summary
st.subheader("📄 Quick Summary")
if st.button("📄 Sinh tóm tắt nhanh"):
    res = requests.get(f"{BACKEND_URL}/summarize/")
    st.success(res.json()["summary"])


