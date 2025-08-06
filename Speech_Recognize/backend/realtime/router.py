from fastapi import APIRouter, UploadFile, File
from inference.preprocessing import preprocess_audio, generate_caption_deploy
from inference.load_model import load_model, load_tokenizer_json, load_tokenizer
from backend.realtime.memory import memory
import torch
import io
import soundfile as sf

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model_path = "D:/Project/Speech_Recognize/model_weights/resnet50_lstm.pth"
tokenizer_path = "D:/Project/Speech_Recognize/model_weights/word2idx.json"

tokenizer = load_tokenizer_json(tokenizer_path)
model = load_model(model_path, tokenizer, device)
router = APIRouter()

#
# @router.post("/stream-audio/")
# async def stream_audio(file: UploadFile = File(...)):
#     try:
#         temp_audio = await file.read()
#         mel = preprocess_audio(temp_audio).to(device)
#         caption = generate_caption_deploy(model, mel, tokenizer, device)
#         memory.add_caption(caption)
#         return {"caption": caption}
#     except Exception as e:
#         return {"error": str(e)}
#
# @router.get("/latest-caption/")
# async def latest_caption():
#     if memory.get_captions():
#         return {"caption": memory.get_captions()[-1]}
#     return {"caption": ""}
#
# @router.get("/summarize/")
# async def summarize():
#     captions = memory.get_captions()
#     if not captions:
#         return {"summary": "Chưa có dữ liệu nào được ghi nhận."}
#
#     full_text = " ".join(captions)
#     summary = f"Tóm tắt nhanh: Trong cuộc hội thoại, người nói đã nói về: {full_text[:300]}..."
#     return {"summary": summary}




@router.post("/stream-audio/")
async def stream_audio(file: UploadFile = File(...)):
    try:
        audio_bytes = await file.read()
        # Chuyển bytes thành mảng numpy
        audio_data, sr = sf.read(io.BytesIO(audio_bytes))
        mel = preprocess_audio(audio_data).to(device)
        caption = generate_caption_deploy(model, mel, tokenizer, device)
        memory.add_caption(caption)
        return {"caption": caption}
    except Exception as e:
        return {"error": str(e)}

@router.get("/latest-caption/")
async def latest_caption():
    if memory.get_captions():
        return {"caption": memory.get_captions()[-1]}
    return {"caption": ""}

@router.get("/history-captions/")
async def history():
    return {"captions": memory.get_captions()}

@router.get("/summarize/")
async def summarize():
    captions = memory.get_captions()
    if not captions:
        return {"summary": "Chưa có dữ liệu nào được ghi nhận."}
    full_text = " ".join(captions)
    summary = f"Tóm tắt nhanh: {full_text[:300]}..."
    return {"summary": summary}

@router.post("/clear/")
async def clear_memory():
    memory.clear()
    return {"message": "Memory cleared."}