import os

# Ưu tiên đọc từ ENV, fallback về localhost
BACKEND_URL = os.getenv("BACKEND_URL", "http://localhost:8000")
