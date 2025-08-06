# Speech Recognize Audio Captioning 🆙️

## 1️⃣ Project Description

* Dự án demo hệ thống nhận dạng audio và sinh mô tả bằng AI.
* Backend: FastAPI
* Frontend: Streamlit
* Model inference: AudioCaptioningModel

---

## 2️⃣ Folder Struture

```bash
.
├── backend/        # FastAPI backend
├── frontend/       # Streamlit frontend
├── inference/      # Model & tokenizer loading
├── model_weights/  # File model, tokenizer
├── training/       # (Optional: Training code)
├── data/           # (Optional: Dataset)
├── docker-compose.yml
├── README.md
```

---

## 3️⃣ Deploy LOCAL (recommended to run local first)

### Bước 1: Create virtual environment

```bash
python -m venv .venv
```

### Bước 2: Activate virtual environment

Windows:

```bash
.\.venv\Scripts\activate
```

Linux/Mac:

```bash
source .venv/bin/activate
```

### Bước 3: Set up dependencies

Backend:

```bash
cd backend
pip install -r requirements.txt
```

Frontend:

```bash
cd ../frontend
pip install -r requirements.txt
```

### Bước 4: Run backend

Back to root directory project:

```bash
cd ../
uvicorn backend.main:app --reload
```

Backend will run at: `http://127.0.0.1:8000`

### Bước 5: Run frontend

Open new terminal:

```bash
cd frontend
streamlit run app.py
```

Frontend will run at: `http://localhost:8501`

---

## 4️⃣ Run with Docker (optional)

### Bước 1: Prepare Docker Compose

```bash
docker-compose up --build
```

After build is complete, access:

* Backend: `http://localhost:8000`
* Frontend: `http://localhost:8501`

### Lưu ý:

* In Docker Compose, backend + frontend are installed with automatic environment.
* Docker cache locally is usually large because it includes images, layers, etc.

---

## 5️⃣ Ghi chú quan trọng

* Model files (.pth), tokenizer.pkl must be placed in the correct location `model_weights/`.
* Files in `inference/` are used for loading models when the backend starts.
* In Docker build, the more dependencies, the first build will be slower (due to caching), the next will be faster.

---


