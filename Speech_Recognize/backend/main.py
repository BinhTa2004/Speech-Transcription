from fastapi import FastAPI, UploadFile, File
import shutil
import torch
import uuid
import os
import threading
import json

from inference.preprocessing import preprocess_audio, generate_caption_deploy, generate_caption_optimized_deploy
from inference.load_model import load_model, load_tokenizer_json, load_tokenizer
from backend.realtime.router import router as realtime_router
import backend.micro_stream as mic_stream

app = FastAPI()
app.include_router(realtime_router)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "D:/Project/Speech_Recognize/model_weights/resnet50_lstm.pth"
tokenizer_path = "D:/Project/Speech_Recognize/model_weights/word2idx.json"

tokenizer = load_tokenizer_json(tokenizer_path)
model = load_model(model_path, tokenizer, device)

os.makedirs("backend/uploads", exist_ok=True)

@app.post("/upload-audio/")
async def upload_audio(file: UploadFile = File(...)):
    temp_filename = f"backend/uploads/{uuid.uuid4()}.wav"
    with open(temp_filename, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    try:
        mel = preprocess_audio(temp_filename).to(device)
        caption = generate_caption_optimized_deploy(model, mel, tokenizer, device=device)
        return {"caption": caption}
    except Exception as e:
        return {"error": str(e)}
    finally:
        os.remove(temp_filename)

@app.post("/start-mic-streaming/")
def start_mic_streaming():
    threading.Thread(
        target=mic_stream.start_streaming,
        args=(model, tokenizer, device),
        daemon=True
    ).start()
    return {"message": "Microphone streaming started."}

@app.post("/stop-mic-streaming/")
def stop_mic_streaming():
    mic_stream.stop_streaming()
    return {"message": "Stopped microphone streaming."}

@app.get("/mic-level/")
async def mic_level():
    return {"level": mic_stream.mic_current_level}

@app.get("/waveform/")
async def get_waveform():
    return {"waveform": mic_stream.waveform_data}

@app.get("/full-summary/")
async def full_summary():
    try:
        captions = []
        with open("meeting_transcript.json", "r") as f:
            for line in f:
                entry = json.loads(line)
                captions.append(entry["caption"])
        full_text = " ".join(captions)
        # Tóm tắt thô (bạn có thể thay thế bằng GPT sau)
        summary = f"Tóm tắt toàn bộ: {full_text[:1000]}..."
        return {"summary": summary}
    except:
        return {"summary": "Chưa có dữ liệu nào."}

# Gõ lệnh để tắt tất cả tiến trình uvicorn:
# Gõ lệnh để tắt tất cả tiến trình python:
# taskkill /IM uvicorn.exe /F
# taskkill /IM python.exe /F
# tasklist | findstr uvicorn
# tasklist | findstr python