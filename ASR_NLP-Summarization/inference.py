from pipeline.asr_summarization_pipeline import SpeechToSummaryPipeline
from audio_streamer.MicrophoneStreamer import RecorderTranscriber

def run_mic_pipeline():
    mic = RecorderTranscriber(sample_rate=16000, chunk_seconds=5)
    pipeline = SpeechToSummaryPipeline()

    print("🎧 Bắt đầu nghe từ mic...")

    try:
        while True:
            # 1) Thu audio từ micro → numpy + file path
            audio_data, filepath = mic.record_chunk()

            # 2) Đưa audio vào pipeline xử lý
            result = pipeline.run(audio_data)

            print("\n=== FILE ===")
            print(filepath)

            print("\n=== TRANSCRIPT ===")
            print(result["transcript"])

            print("\n=== SUMMARY ===")
            print(result["summary"])
            print("===============================")

    except KeyboardInterrupt:
        print("\n🛑 Dừng.")

if __name__ == "__main__":
    run_mic_pipeline()
