# import sounddevice as sd
# import numpy as np
# import requests
# import time
# import queue
# import io
# import soundfile as sf
# import threading
#
# BACKEND_URL = "http://localhost:8000/stream-audio/"
# q = queue.Queue()
# current_volume = 0
#
# def audio_callback(indata, frames, time_info, status):
#     if status:
#         print(status)
#     q.put(indata.copy())
#
# def send_audio_stream(model, tokenizer, device):
#     buffer = []
#     window_duration = 2  # seconds
#     sample_rate = 16000
#     window_size = sample_rate * window_duration
#
#     while True:
#         data = q.get()
#         buffer.append(data)
#         total_len = sum([len(chunk) for chunk in buffer])
#
#         if total_len >= window_size:
#             audio_data = np.concatenate(buffer)[:window_size]
#             buffer = [np.concatenate(buffer)[window_size:]]
#
#             with io.BytesIO() as f:
#                 sf.write(f, audio_data, samplerate=sample_rate, format='WAV')
#                 f.seek(0)
#                 files = {'file': ('chunk.wav', f, 'audio/wav')}
#                 try:
#                     response = requests.post(BACKEND_URL, files=files, timeout=10)
#                     print("Sent chunk, caption:", response.json().get("caption"))
#                 except Exception as e:
#                     print("Error sending stream:", e)
#
# def start_streaming(model, tokenizer, device):
#     stream_thread = threading.Thread(target=send_audio_stream, args=(model, tokenizer, device))
#     stream_thread.start()
#
#     with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
#         while True:
#             time.sleep(0.1)
#
#
# # Hàm trả về mức volume hiện tại (cho frontend gọi)
# def get_current_volume():
#     return current_volume
#
#

#
# import sounddevice as sd
# import numpy as np
# import requests
# import io
# import soundfile as sf
# import queue
# import time
# from backend.realtime.memory import memory
# from inference.preprocessing import preprocess_audio, generate_caption_deploy
#
# BACKEND_STREAM_URL = "http://localhost:8000/stream-audio/"
# q = queue.Queue()
#
# running = False
#
# def audio_callback(indata, frames, time, status):
#     if status:
#         print(status)
#     q.put(indata.copy())
#
# def start_streaming(model, tokenizer, device):
#     global running
#     running = True
#
#     stream = sd.InputStream(callback=audio_callback, channels=1, samplerate=16000)
#     stream.start()
#
#     buffer = []
#     window_size = 16000 * 2  # 2 seconds
#
#     while running:
#         if not q.empty():
#             data = q.get()
#             buffer.append(data)
#             total_len = sum(len(chunk) for chunk in buffer)
#
#             if total_len >= window_size:
#                 audio_data = np.concatenate(buffer)[:window_size]
#                 buffer = [np.concatenate(buffer)[window_size:]]
#
#                 with io.BytesIO() as f:
#                     sf.write(f, audio_data, samplerate=16000, format='WAV')
#                     f.seek(0)
#                     mel = preprocess_audio(f.read()).to(device)
#                     caption = generate_caption_deploy(model, mel, tokenizer, device)
#                     memory.add_caption(caption)
#                     print("Realtime caption:", caption)
#
#     stream.stop()
#     stream.close()
#
# def stop_streaming():
#     global running
#     running = False




import sounddevice as sd
import numpy as np
import requests
import time
import queue
import threading
import io
import soundfile as sf
import json
from datetime import datetime

BACKEND_URL = "http://localhost:8000/stream-audio/"
q = queue.Queue()
streaming = False
mic_current_level = 0
waveform_data = []
transcript_file = "meeting_transcript.json"

def save_transcript(caption):
    timestamp = datetime.now().isoformat()
    entry = {"time": timestamp, "caption": caption}

    try:
        with open(transcript_file, "a") as f:
            json.dump(entry, f)
            f.write("\n")
    except Exception as e:
        print("Save transcript error:", e)

def audio_callback(indata, frames, time_info, status):
    if status:
        print(status)
    q.put(indata.copy())

def send_audio_stream():
    buffer = []
    window_duration = 2  # seconds
    sample_rate = 16000
    window_size = sample_rate * window_duration

    global waveform_data

    while streaming:
        try:
            data = q.get(timeout=1)
            buffer.append(data)
            total_len = sum([len(chunk) for chunk in buffer])

            if total_len >= window_size:
                audio_data = np.concatenate(buffer)[:window_size]
                buffer = [np.concatenate(buffer)[window_size:]]

                # Convert to bytes (wav format)
                with io.BytesIO() as f:
                    sf.write(f, audio_data, samplerate=sample_rate, format='WAV')
                    f.seek(0)
                    files = {'file': ('chunk.wav', f, 'audio/wav')}
                    response = requests.post(BACKEND_URL, files=files, timeout=10)

                    caption = response.json().get("caption")
                    print("Caption:", caption)

                    # Lưu transcript
                    if caption:
                        save_transcript(caption)

                # Visualize amplitude level:
                volume_norm = np.linalg.norm(audio_data) * 10
                bars = "#" * int(volume_norm)
                print(f"Mic Level: {bars}")

                waveform_data = audio_data.flatten().tolist()

        except queue.Empty:
            continue
        except Exception as e:
            print("Error sending stream:", e)

def start_streaming():
    global streaming
    streaming = True

    threading.Thread(target=send_audio_stream, daemon=True).start()

    with sd.InputStream(callback=audio_callback, channels=1, samplerate=16000):
        while streaming:
            time.sleep(0.1)

def stop_streaming():
    global streaming
    streaming = False
