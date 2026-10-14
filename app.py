"""
app.py —— 语音模仿相似度比对 · 网页版后端
Flask 提供静态页面 + /compare 接口：
  前端把"原音频"和"模仿录音"都转成 WAV 上传，
  后端直接交给 voice_compare.compare 计算相似度并返回 JSON。

运行：
  /Users/zichen/.workbuddy/binaries/python/envs/voicecmp/bin/python app.py
然后浏览器打开 http://localhost:5000
"""

import os
import uuid
from flask import Flask, request, jsonify, Response
import voice_compare

app = Flask(__name__)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = "/tmp/voicecmp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


@app.route("/")
def index():
    with open(os.path.join(BASE_DIR, "index.html"), "r", encoding="utf-8") as f:
        return Response(f.read(), mimetype="text/html")


@app.route("/ping")
def ping():
    """前端自检用：能返回说明后端在运行。"""
    return jsonify({"ok": True})


@app.route("/compare", methods=["POST"])
def compare():
    original = request.files.get("original")
    mimic = request.files.get("mimic")
    if not original or not mimic:
        return jsonify({"error": "缺少音频文件，请同时上传原音频和模仿录音"}), 400

    opath = os.path.join(UPLOAD_DIR, uuid.uuid4().hex + ".wav")
    mpath = os.path.join(UPLOAD_DIR, uuid.uuid4().hex + ".wav")
    original.save(opath)
    mimic.save(mpath)

    try:
        result = voice_compare.compare(opath, mpath)
    except Exception as e:  # 算法或解码出错时返回可读错误
        return jsonify({"error": f"比对失败：{e}"}), 500
    finally:
        for p in (opath, mpath):
            try:
                os.remove(p)
            except OSError:
                pass

    return jsonify(result)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=True)
