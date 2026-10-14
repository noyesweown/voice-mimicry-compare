"""
语音模仿相似度比对 —— Web 后端 (Flask)
主分数用预训练声纹嵌入模型(resemblyzer)计算两段音频的"声音相似度"；
同时用 v2 整句级启发式给出 音色/语调/能量 分项作为补充参考。

运行:
    python app.py
浏览器打开 http://localhost:5001

部署: 适配 HuggingFace Spaces / Render (读 $PORT)，见 README。
"""
import os
import tempfile
from flask import Flask, request, jsonify, send_from_directory
import numpy as np
import soundfile as sf

import voice_embed as ve
import voice_compare_v2 as v2

app = Flask(__name__, static_folder=".", template_folder=".")
PORT = int(os.environ.get("PORT", 5001))


def save_upload(f):
    """把上传的文件存成临时 wav 并返回路径。"""
    suffix = ".wav"
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    f.save(path)
    return path


@app.route("/")
def index():
    return send_from_directory(".", "index.html")


@app.route("/ping")
def ping():
    return jsonify(ok=True)


@app.route("/compare", methods=["POST"])
def compare():
    try:
        o = request.files["original"]
        m = request.files["mimic"]
        op = save_upload(o)
        mp = save_upload(m)
        y1 = ve.load_audio(op)
        y2 = ve.load_audio(mp)
        # 主分数: 声纹嵌入余弦相似度
        total = ve.similarity(y1, y2)
        # 分项: v2 整句级启发式(仅供參考)
        comp = v2.compare(y1, y2)
        return jsonify({
            "total": total,
            "mfcc": comp["mfcc"],
            "pitch": comp["pitch"],
            "energy": comp["energy"],
        })
    except Exception as e:
        return jsonify(error=str(e)), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=PORT, debug=True)
