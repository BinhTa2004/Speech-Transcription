import torch
import whisper

class WhisperASR:
    def __init__(self, model_name="small"):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.model = whisper.load_model(model_name, device=self.device)

    def transcribe(self, audio_path: str) -> str:
        result = self.model.transcribe(audio_path, fp16=False)
        return result["text"]
