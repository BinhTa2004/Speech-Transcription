# Real-time Speech Transcription and Summarization System

An end-to-end system that supports **real-time audio captioning**, **speech recognition**, and **summarization**, designed for intelligent audio understanding applications. Powered by state-of-the-art deep learning models (ResNet, LSTM, Whisper, Wav2Vec2, PEGASUS, T5), deployed with FastAPI + Streamlit + Docker.

---

## 🎯 Use Cases
- 🖼️ **Audio Captioning**: Generate natural-language descriptions of arbitrary sounds (e.g., "a dog barking", "applause", "car engine revving").
- 🗣️ **Speech Transcription**: Convert human speech to text using ASR models (Whisper, Wav2Vec2).
- ✂️ **Summarization**: Summarize long-form spoken content into concise text with T5 / PEGASUS.

---

## 🛠️ Technologies
- Python, FastAPI, Streamlit
- PyTorch, TorchAudio, Transformers (HuggingFace)
- ASR: Whisper, Wav2Vec2
- Summarizer: PEGASUS, T5
- Audio Captioning: ResNet + LSTM
- Docker, Docker Compose

---

## 🔧 Quick Setup
### 1️⃣ Clone repo & setup env
```bash
git clone https://github.com/BinhTa2004/Speech-Transcription.git
cd Speech-Transcription

python -m venv .venv
source .venv/bin/activate  # Windows: .\.venv\Scripts\activate
```

### 2️⃣ Install dependencies
```bash
pip install -r requirements.txt
```
### 3️⃣ Run backend (FastAPI)
```bash
uvicorn backend.main:app --reload
```
### 4️⃣ Run frontend (Streamlit)
```bash
cd frontend
streamlit run app.py
```
### 🐳 Docker (optional)
```bash
docker-compose up --build
```
⚠️ Notes:
* The first build may take several minutes as Docker installs models and dependencies.
* Docker images and layers can consume significant disk space (especially with deep learning models).
