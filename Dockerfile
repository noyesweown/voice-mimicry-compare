FROM python:3.11-slim

WORKDIR /app

# libsndfile 供 soundfile 读取 wav；ffmpeg 备用（前端已把音频转成 wav，通常不需要）
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

ENV PORT=7860
EXPOSE 7860
CMD ["python", "app.py"]
