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

## 给同事/他人使用
- **GitHub 共享代码**：对方 clone 后按上面步骤本地运行。
- **零安装给远程同事用**：用内网穿透（如 `ngrok http 5001`）把你的本地服务暴露成公网链接发给他；或部署到免费托管（Render / Hugging Face Spaces）拿到固定网址。

## 文件
- `app.py`：Flask 后端（页面 + /compare 接口）
- `index.html`：前端页面（上传 + 浏览器录音 + 结果可视化）
- `voice_compare.py`：核心比对算法
- `demo_test.py`：用合成音频自测
- `record_and_compare.py`：命令行的"录音→比对"入口（需 `pip install sounddevice`）
