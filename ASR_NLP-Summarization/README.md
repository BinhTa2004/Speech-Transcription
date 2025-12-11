## 1) Mở terminal 1 → chạy server FastAPI

```bash
python server.py
```

* Server chạy tại: http://0.0.0.0:8000
* WebSocket chạy tại: ws://localhost:8000/ws

## 2) Mở terminal 2 → chạy app Streamlit

```bash
streamlit run app.py
```

* Giao diện mở tại: http://localhost:8501

## 3) Nói vào micro → Streamlit hiển thị

* Transcript (văn bản nói real-time)
* Summary (tóm tắt)

## 4) Cài thư viện cần thiết

Nếu chưa cài:
```bash
pip install fastapi uvicorn websockets streamlit sounddevice numpy
```
