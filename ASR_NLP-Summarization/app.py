# app.py
import streamlit as st
import asyncio
import websockets
import json

st.set_page_config(page_title="Real-time ASR + Summary", layout="wide")

st.title("🎙️ Real-time Speech Transcription & Summarization")

transcript_box = st.empty()
summary_box = st.empty()
file_box = st.empty()

async def ws_client():
    uri = "ws://localhost:8000/ws"
    async with websockets.connect(uri) as ws:
        while True:
            message = await ws.recv()
            data = json.loads(message)

            transcript_box.markdown(f"### 📝 Transcript\n```\n{data['transcript']}\n```")
            summary_box.markdown(f"### 🔍 Summary\n```\n{data['summary']}\n```")
            file_box.markdown(f"📁 **Audio saved:** `{data['file_path']}`")


def start_ws():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(ws_client())


st.button("▶ Start Listening", on_click=start_ws)
