# 语音模仿相似度比对 (Voice Mimicry Similarity)

上传一段原音频，对着麦克风模仿它，程序给出模仿与原音频的相似度（音色 / 音高 / 能量 / 综合）。

## 原理
- 预处理：统一重采样到 16kHz 单声道、去直流、音量归一。
- **主分数（声纹相似度）**：用预训练说话人嵌入模型 **resemblyzer** 把每段音频编码成 512 维"声纹向量"，两段向量的**余弦相似度**就是"像不像同一个人"的分数（映射到 0~1）。模型在海量说话人数据上训练，零样本、无需标定、不依赖样例。
- 分项（参考）：整句级 MFCC 均值/标准差（音色）、基频轮廓（语调）、能量轮廓（节奏）各算一次余弦相似度，作为细节参考。

> 早期版本用逐帧 MFCC+DTW 的启发式算法，对真实语音（辅音、呼吸、噪声）极其敏感，会算出 0%。现版本改用声纹嵌入模型，对真实人声稳健得多。

## 运行（本地）
```bash
pip install -r requirements.txt
python app.py
```
浏览器打开 http://localhost:5001
1. 上传原音频（wav/mp3/m4a…）
2. 点"开始录音"对着麦克风模仿，再点一次停止
3. 点"开始比对"查看分数

> 注意：必须用 `http://localhost:5001` 打开页面，不要直接双击 index.html（否则前端连不上后端）。首次运行会自动下载约 25MB 的 resemblyzer 模型权重。

## 部署给远程同事用
声纹模型必须跑在 Python 后端，所以需要一个能跑 Docker / Python 的托管：

### 方案 A：Render（免费，推荐）
1. 登录 render.com → 右上角 **New** → **Web Service** → 连你的 GitHub 仓库 `noyesweown/voice-mimicry-compare`。
2. 在 "Runtime" 区域 Render 会自动检测到仓库里的 **`render.yaml`**，直接选它即可（无需手动填命令）；若手动建：Runtime 选 **Python**，Build Command `pip install -r requirements.txt`，Start Command `python app.py`。
3. 点 **Create Web Service**，等构建完成（首次约 2–5 分钟，会装 tensorflow 等较重依赖）。
4. 部署完成后页面顶部变绿，给一个 `https://voice-mimicry-compare.onrender.com` 固定网址 → 发给同事，他打开即可上传 + 录音 + 比对，零安装。
5. 免费版特性：15 分钟无人访问会休眠，唤醒需约 1 分钟冷启动；首次请求会下载约 25MB 模型权重（之后缓存在磁盘更快）。

### 方案 B：Hugging Face Spaces（需付费计划）
HF 免费版已不支持 Docker SDK，若要用 Spaces 需升级到付费计划，再选 Docker SDK 并连本仓库。

### 方案 C：本地 + ngrok（临时演示）
本机 `python app.py` 后，用 `ngrok http 5001` 生成公网链接发给同事（需本机保持开机）。

> 前端会在浏览器里把上传文件和录音都转成 WAV 再上传，后端只收 WAV，因此托管环境不需要额外音频解码依赖。

## 文件
- `app.py`：Flask 后端（页面 + /compare 接口，调用声纹模型）
- `index.html`：前端页面（上传 + 浏览器录音 + 结果可视化）
- `voice_embed.py`：声纹嵌入模型封装（resemblyzer）
- `voice_compare_v2.py`：整句级启发式分项（MFCC/音高/能量），作为参考
- `requirements.txt` / `Dockerfile`：依赖与部署配置
