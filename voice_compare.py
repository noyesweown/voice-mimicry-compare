"""
voice_compare.py
语音模仿相似度比对 —— 核心算法库（传统 MFCC + DTW 基线版）

思路：
  1. 把两段音频统一成 16kHz 单声道（特征对采样率敏感）
  2. 提取特征：MFCC（音色+频谱包络）+ 音高F0（旋律感）+ 能量（响度节奏）
  3. 用 DTW（动态时间规整）把不同语速/节奏的两段对齐
  4. 把对齐后的"距离"换算成 0~1 的相似度分数
  5. 多指标加权融合成总分

依赖：numpy scipy librosa
用法：
  from voice_compare import compare
  score = compare("original.wav", "mimic.wav")
  print(score["total"])   # 0~1

说明：下面的 MFCC_SIM_TEMP / PITCH_SIM_TEMP / ENERGY_SIM_TEMP 与 WEIGHTS
是"可调超参"，用真实录音标定效果最好（见 README 说明）。
"""

import numpy as np
import librosa
from scipy.spatial.distance import cdist


TARGET_SR = 16000          # 统一采样率
N_MFCC = 20                # MFCC 维数
MFCC_SIM_TEMP = 0.25       # MFCC 余弦距离→相似度的温度（越大越宽松）
PITCH_SIM_TEMP = 0.18      # 音高距离→相似度的温度
ENERGY_SIM_TEMP = 0.10     # 能量距离→相似度的温度
WEIGHTS = {                # 多指标融合权重，和应为 1
    "mfcc": 0.40,
    "pitch": 0.40,
    "energy": 0.20,
}


def load_audio(path, target_sr=TARGET_SR):
    """读取音频并统一成单声道、目标采样率。返回 (波形y, 采样率sr)。"""
    y, sr = librosa.load(path, sr=target_sr, mono=True)
    y = y - np.mean(y)                 # 去直流
    peak = np.max(np.abs(y))
    if peak > 1e-6:
        y = y / peak                  # 音量归一，削弱设备/麦克风差异
    return y, sr


def extract_mfcc(y, sr, n_mfcc=N_MFCC):
    """提取 MFCC 特征，形状 (n_frames, n_mfcc)，并做帧内均值归一。"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc).T
    mfcc = mfcc - mfcc.mean(axis=0, keepdims=True)   # 削绝对音高、留音色形状
    return mfcc


def extract_prosody(y, sr):
    """提取韵律特征：音高 F0 与能量 RMS（均按帧）。"""
    f0, _, _ = librosa.pyin(
        y, fmin=librosa.note_to_hz("C2"), fmax=librosa.note_to_hz("C6"), sr=sr
    )
    rms = librosa.feature.rms(y=y)[0]
    if len(f0) != len(rms):           # 帧数对齐，方便后续比较
        f0 = librosa.resample(f0.astype(float), orig_sr=len(f0), target_sr=len(rms))
    return f0, rms


def dtw_distance(a, b, metric="euclidean", window_ratio=0.2):
    """
    动态时间规整，返回归一化平均帧距离。
    a, b: (n, d) / (m, d) 的特征序列（一维会自动升维）。
    metric='cosine' 适合 MFCC 等多维向量；'euclidean' 适合已归一化的 1D 序列。
    用 Sakoe-Chiba 带宽约束加速并限制过度扭曲。
    """
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    if b.ndim == 1:
        b = b.reshape(-1, 1)
    if metric == "cosine":
        na = np.linalg.norm(a, axis=1, keepdims=True)
        nb = np.linalg.norm(b, axis=1, keepdims=True)
        a = a / (na + 1e-9)
        b = b / (nb + 1e-9)
        D = cdist(a, b, metric="cosine")
    else:
        D = cdist(a, b, metric="euclidean")

    n, m = D.shape
    INF = np.inf
    cost = np.full((n + 1, m + 1), INF)
    cost[0, 0] = 0.0
    band = int(max(n, m) * window_ratio) + 1
    for i in range(1, n + 1):
        for j in range(max(1, i - band), min(m, i + band) + 1):
            cost[i, j] = D[i - 1, j - 1] + min(cost[i - 1, j],
                                               cost[i, j - 1],
                                               cost[i - 1, j - 1])
    # 回溯路径长度，用于归一化
    i, j, length = n, m, 0
    while i > 0 or j > 0:
        length += 1
        if i > 0 and (j == 0 or (cost[i - 1, j] <= cost[i, j - 1]
                                 and cost[i - 1, j] <= cost[i - 1, j - 1])):
            i -= 1
        elif j > 0 and (i == 0 or cost[i, j - 1] <= cost[i - 1, j - 1]):
            j -= 1
        else:
            i, j = i - 1, j - 1
    return cost[n, m] / max(length, 1)


def _fill_nan(v):
    """NaN 用线性插值填充（F0 的静音帧），全 NaN 则返回零。"""
    v = np.asarray(v, dtype=float)
    if np.isnan(v).all():
        return np.zeros_like(v)
    idx = np.where(~np.isnan(v))[0]
    return np.interp(np.arange(len(v)), idx, v[~np.isnan(v)])


def _zscore(v):
    v = (v - v.mean()) / (v.std() + 1e-9)
    return v


def contour_similarity(f0_ref, f0_tgt, temp=PITCH_SIM_TEMP):
    """
    音高相似度：先把 F0 转成对数(半音尺度)，再 z-score（只比"旋律形状"而非绝对音高），
    最后 DTW 对齐求距离并换算成相似度。
    """
    r = _zscore(np.log2(_fill_nan(f0_ref) + 1e-3))
    t = _zscore(np.log2(_fill_nan(f0_tgt) + 1e-3))
    dist = dtw_distance(r, t, metric="euclidean")
    return float(np.clip(np.exp(-dist / temp), 0.0, 1.0))


def energy_similarity(rms_ref, rms_tgt, temp=ENERGY_SIM_TEMP):
    """能量(响度节奏)相似度：z-score 后 DTW 对齐。"""
    r = _zscore(_fill_nan(rms_ref))
    t = _zscore(_fill_nan(rms_tgt))
    dist = dtw_distance(r, t, metric="euclidean")
    return float(np.clip(np.exp(-dist / temp), 0.0, 1.0))


def mfcc_similarity(mfcc_ref, mfcc_tgt, temp=MFCC_SIM_TEMP):
    """音色/频谱相似度：用余弦距离的 DTW（对向量幅度不敏感）。"""
    dist = dtw_distance(mfcc_ref, mfcc_tgt, metric="cosine")
    return float(np.clip(np.exp(-dist / temp), 0.0, 1.0))


def compare(original_path, mimic_path):
    """主函数：输入两段音频路径，返回各分项与总分（0~1）。"""
    y1, sr1 = load_audio(original_path)
    y2, sr2 = load_audio(mimic_path)

    mfcc1 = extract_mfcc(y1, sr1)
    mfcc2 = extract_mfcc(y2, sr2)
    s_mfcc = mfcc_similarity(mfcc1, mfcc2)

    f0_1, rms_1 = extract_prosody(y1, sr1)
    f0_2, rms_2 = extract_prosody(y2, sr2)
    s_pitch = contour_similarity(f0_1, f0_2)
    s_energy = energy_similarity(rms_1, rms_2)

    total = (WEIGHTS["mfcc"] * s_mfcc
             + WEIGHTS["pitch"] * s_pitch
             + WEIGHTS["energy"] * s_energy)

    return {
        "mfcc_similarity": round(s_mfcc, 4),
        "pitch_similarity": round(s_pitch, 4),
        "energy_similarity": round(s_energy, 4),
        "total": round(float(total), 4),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("用法: python voice_compare.py 原音频.wav 模仿音频.wav")
        sys.exit(1)
    res = compare(sys.argv[1], sys.argv[2])
    print(f"音色相似度 : {res['mfcc_similarity']:.3f}")
    print(f"音高相似度 : {res['pitch_similarity']:.3f}")
    print(f"能量相似度 : {res['energy_similarity']:.3f}")
    print(f"综合相似度 : {res['total']:.3f}  ({res['total']*100:.1f}%)")
