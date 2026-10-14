"""
语音模仿相似度比对 v2 —— 整句级特征（utterance-level）

相比 v1（逐帧 MFCC+DTW），v2 用更稳定、人类可解释的方案：
  1. 音色(timbre) : 整句 MFCC 的 均值+标准差(26维) -> 余弦相似度
  2. 语调(pitch)  : 两段基频轮廓重采样到等长 -> z-score -> 余弦相似度
  3. 能量(energy) : 两段能量轮廓重采样到等长 -> z-score -> 余弦相似度

所有相似度都落在 [-1,1] 区间的余弦上，再映射到 [0,1]。
无需脆弱的温度系数，对真实语音稳健得多。

依赖: numpy, scipy, librosa
"""
import numpy as np
import librosa

SR = 16000
N_MFCC = 13
CONTOUR_POINTS = 100  # 音高/能量轮廓重采样点数


def load_audio(path, sr=SR):
    y, _ = librosa.load(path, sr=sr, mono=True)
    y = y - y.mean()
    peak = np.max(np.abs(y)) + 1e-9
    y = y / peak  # 音量归一，避免录音电平差异主导结果
    return y.astype(np.float32)


def utterance_mfcc_vec(y, sr=SR):
    """返回 (26,) : 13维MFCC的均值+标准差。"""
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=N_MFCC, n_fft=2048, hop_length=512,
                                n_mels=26, fmax=8000)
    # (13, T) -> mean/std -> (26,)
    return np.concatenate([mfcc.mean(axis=1), mfcc.std(axis=1)])


def f0_contour(y, sr=SR):
    """基频轮廓，NaN 用线性插值补全。"""
    f0 = librosa.pyin(y, fmin=80, fmax=500, sr=sr, frame_length=2048, hop_length=512)[0]
    f0 = np.asarray(f0, dtype=np.float64)
    nans = np.isnan(f0)
    if nans.all():
        return np.zeros(CONTOUR_POINTS)
    # 对数域插值更稳
    f0log = np.where(nans, np.nan, np.log2(f0 + 1e-3))
    idx = np.where(~nans)[0]
    f0log = np.interp(np.arange(len(f0)), idx, f0log[idx])
    f0 = 2 ** f0log
    return f0


def rms_contour(y, sr=SR):
    rms = librosa.feature.rms(y=y, frame_length=2048, hop_length=512)[0]
    return np.asarray(rms, dtype=np.float64)


def resample_contour(c, n=CONTOUR_POINTS):
    """把任意长度轮廓线性重采样到 n 点。"""
    if len(c) == 0:
        return np.zeros(n)
    old = np.linspace(0, 1, len(c))
    new = np.linspace(0, 1, n)
    return np.interp(new, old, c)


def zscore(x):
    x = np.asarray(x, dtype=np.float64)
    m = x.mean()
    s = x.std() + 1e-9
    return (x - m) / s


def cosine(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    na = np.linalg.norm(a) + 1e-9
    nb = np.linalg.norm(b) + 1e-9
    return float((a / na).dot(b / nb))


def clip01(x):
    return float(min(1.0, max(0.0, x)))


def compare(original, mimic, sr=SR):
    """original/mimic: 一维 float 音频(已load_audio)。返回各分项与综合相似度。"""
    # 音色
    va = utterance_mfcc_vec(original, sr)
    vb = utterance_mfcc_vec(mimic, sr)
    mfcc_sim = clip01((cosine(va, vb) + 1) / 2)

    # 语调：对数域 -> 重采样 -> z-score -> 余弦
    fa = resample_contour(f0_contour(original, sr))
    fb = resample_contour(f0_contour(mimic, sr))
    pitch_sim = clip01((cosine(zscore(np.log2(fa + 1e-3)), zscore(np.log2(fb + 1e-3))) + 1) / 2)

    # 能量：重采样 -> z-score -> 余弦
    ea = resample_contour(rms_contour(original, sr))
    eb = resample_contour(rms_contour(mimic, sr))
    energy_sim = clip01((cosine(zscore(ea), zscore(eb)) + 1) / 2)

    total = 0.5 * mfcc_sim + 0.3 * pitch_sim + 0.2 * energy_sim
    return {
        "mfcc": round(mfcc_sim, 3),
        "pitch": round(pitch_sim, 3),
        "energy": round(energy_sim, 3),
        "total": round(total, 3),
    }


if __name__ == "__main__":
    import sys
    if len(sys.argv) < 3:
        print("用法: python voice_compare_v2.py 原音频.wav 模仿音频.wav")
        sys.exit(1)
    a = load_audio(sys.argv[1])
    b = load_audio(sys.argv[2])
    print("相似度:", compare(a, b))
