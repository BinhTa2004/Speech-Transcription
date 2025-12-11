# import os
# import datetime
# import numpy as np
# import sounddevice as sd
# import soundfile as sf
#
# class RecorderTranscriber:
#     def __init__(self, save_dir="recordings", sample_rate=16000, chunk_seconds=3):
#         self.save_dir = save_dir
#         self.sample_rate = sample_rate
#         self.chunk_seconds = chunk_seconds
#         self.chunk_samples = self.sample_rate * self.chunk_seconds
#
#         os.makedirs(self.save_dir, exist_ok=True)
#
#     def record_chunk(self):
#         """Ghi 1 chunk audio và trả về numpy array."""
#         print("🎙 Đang ghi 1 chunk...", flush=True)
#         audio = sd.rec(
#             self.chunk_samples,
#             samplerate=self.sample_rate,
#             channels=1,
#             dtype=np.float32,
#         )
#         sd.wait()
#         return audio.flatten()
#
#     def save_audio(self, audio):
#         """Lưu WAV nếu cần."""
#         filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
#         file_path = os.path.join(self.save_dir, filename)
#         sf.write(file_path, audio, self.sample_rate)
#         return file_path



import os
import datetime
import numpy as np
import sounddevice as sd
import soundfile as sf
from pipeline.asr_summarization_pipeline import SpeechToSummaryPipeline

class RecorderTranscriber:
    def __init__(self, save_dir="recordings", sample_rate=16000, chunk_seconds=30):
        self.save_dir = save_dir
        self.sample_rate = sample_rate
        self.chunk_seconds = chunk_seconds
        self.chunk_samples = self.sample_rate * self.chunk_seconds

        os.makedirs(self.save_dir, exist_ok=True)

    def record_chunk(self):
        """Ghi 1 chunk audio và trả về numpy array."""
        print("🎙 Đang ghi 1 chunk...", flush=True)
        audio = sd.rec(
            self.chunk_samples,
            samplerate=self.sample_rate,
            channels=1,
            dtype=np.float32,
        )
        sd.wait()

        filename = datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".wav"
        file_path = os.path.join(self.save_dir, filename)
        sf.write(file_path, audio, self.sample_rate)
        print(f"💾 Đã lưu file: {file_path}")

        return audio.flatten(),file_path


class MicPipeline:
    def __init__(self):
        self.pipeline = SpeechToSummaryPipeline()
        self.recorder = RecorderTranscriber()

    def process_once(self):
        audio, path = self.recorder.record_chunk()
        result = self.pipeline.run(audio)
        return result, path