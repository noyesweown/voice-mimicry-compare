# 语音模仿相似度比对 (Voice Mimicry Similarity)

上传一段原音频，对着麦克风模仿它，程序给出模仿与原音频的相似度（音色 / 音高 / 能量 / 综合）。

## 原理
- 预处理：统一重采样到 16kHz 单声道、去直流、音量归一。
- 特征：MFCC（音色/频谱包络）、F0 基频（音高/语调）、RMS 能量（响度节奏）。
- 对齐：DTW 动态时间规整，消除语速/节奏差异。
- 融合：各特征距离换算成 0~1 相似度后加权融合。

## 运行（本地）
```bash
pip install -r requirements.txt
python app.py
```
浏览器打开 http://localhost:5001
1. 上传原音频（wav/mp3/m4a…）
2. 点"开始录音"对着麦克风模仿，再点一次停止
3. 点"开始比对"查看分数

> 注意：必须用 `http://localhost:5001` 打开页面，不要直接双击 index.html（否则前端连不上后端）。

## 部署到免费托管（让远程同事点链接即用）
### Hugging Face Spaces（推荐，免费、固定网址）
1. 登录 huggingface.co → New Space → 选 **Docker** SDK，可见性选 Public（或 Private）。
2. 把本仓库内容（含 `Dockerfile`）推到 Space 的 git 仓库，或直接连 GitHub 仓库。
3. Space 会自动构建镜像、监听 `$PORT`（默认 7860），构建完成后给一个 `*.hf.space` 网址发给同事即可。
4. 同事打开网址 → 上传原音频 → 浏览器录音 → 比对，全部在网页完成，无需安装任何东西。

### Render（备选）
1. 登录 render.com → New Web Service → 连 GitHub 仓库。
2. Build Command: `pip install -r requirements.txt`，Start Command: `python app.py`。
3. Render 会自动注入 `PORT` 环境变量，部署后给固定网址。

> 前端会在浏览器里把上传文件和录音都转成 WAV 再上传，后端只收 WAV，因此托管环境不需要额外音频解码依赖。

## 文件
- `app.py`：Flask 后端（页面 + /compare 接口）
- `index.html`：前端页面（上传 + 浏览器录音 + 结果可视化）
- `voice_compare.py`：核心比对算法
- `demo_test.py`：用合成音频自测
- `record_and_compare.py`：命令行的"录音→比对"入口（需 `pip install sounddevice`）
