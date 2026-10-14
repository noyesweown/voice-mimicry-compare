"""
声纹嵌入相似度 —— 用预训练说话人嵌入模型(resemblyzer)直接比"两段音频像不像同一个人"。

相比 v1/v2 的启发式特征，嵌入模型在海量说话人数据上预训练，零样本、无需标定。
流程: 每段音频 -> 512维声纹向量(d-vector) -> 余弦相似度 -> 映射到[0,1]。
依赖: resemblyzer (会自动下载约25MB模型权重), librosa
"""
import numpy as np
import librosa
from resemblyzer import VoiceEncoder, preprocess_wav

SR = 16000
_encoder = None


def _get_encoder():
    global _encoder
    if _encoder is None:
        _encoder = VoiceEncoder()
    return _encoder


def load_audio(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = y - y.mean()
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak).astype(np.float32)


def embed(audio, sr=SR):
    """audio: 路径(str) 或 一维float数组。返回512维向量。"""
    if isinstance(audio, str):
        wav = load_audio(audio, sr)
    else:
        # 已是数组：重采样到16k并归一
        if sr != SR:
            audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=SR)
        audio = np.asarray(audio, dtype=np.float32)
        audio = audio - audio.mean()
        peak = np.max(np.abs(audio)) + 1e-9
        wav = audio / peak
    return _get_encoder().embed_utterance(wav)


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    return float(a.dot(b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-9))


def similarity(original, mimic, sr=SR):
    """返回 0~1 相似度。original/mimic 可为路径或数组。"""
    ea = embed(original, sr)
    eb = embed(mimic, sr)
    cos = cosine(ea, eb)
    return round((cos + 1) / 2, 3)  # 余弦[-1,1] -> [0,1]


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python voice_embed.py 原音频.wav 模仿音频.wav")
        sys.exit(1)
    print("相似度:", similarity(sys.argv[1], sys.argv[2]))
