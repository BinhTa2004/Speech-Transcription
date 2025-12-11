from ASR.whisper_inference import WhisperASR
from NLP.summarizer import TextSummarizer

class SpeechToSummaryPipeline:
    def __init__(self):
        print("Loading Whisper ASR...")
        self.asr = WhisperASR(model_name="small")

        print("Loading T5 Summarizer...")
        self.summarizer = TextSummarizer(model_name="t5-small")

    def run(self, audio_path: str) -> dict:
        transcript = self.asr.transcribe(audio_path)
        summary = self.summarizer.summarize(transcript)

        return {
            "transcript": transcript,
            "summary": summary
        }
