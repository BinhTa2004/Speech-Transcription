# server.py
import uvicorn
import asyncio
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

from audio_streamer.MicrophoneStreamer import MicPipeline

app = FastAPI()
mic = MicPipeline()

@app.websocket("/ws")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()
    print("🔌 Client connected")

    try:
        while True:
            # Lấy 1 chunk từ microphone + pipeline xử lý
            result, path = mic.process_once()

            await websocket.send_json({
                "transcript": result["transcript"],
                "summary": result["summary"],
                "file_path": path
            })

            await asyncio.sleep(0.01)

    except WebSocketDisconnect:
        print("❌ Client disconnected")

    except Exception as e:
        print("⚠ Server error:", e)

if __name__ == "__main__":
    uvicorn.run("server:app", host="0.0.0.0", port=8000, reload=False)
