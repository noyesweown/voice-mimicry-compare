"""
demo_test.py
不依赖真实录音，用合成音频验证 voice_compare 整条链路：
  - 同一段旋律 A（自己比自己，应≈1.0）
  - A 的"模仿版"A'（加抖动+噪声+变速，应较高）
  - 完全无关的旋律 C（应明显更低）
生成的中间 wav 写在 /tmp 下，不污染工作区。
"""

import numpy as np
import soundfile as sf
import os
from voice_compare import compare

SR = 16000


def make_melody(contour, vib=4.0, noise=0.0, speed=1.0, seed=0):
    """根据音高轮廓 contour(归一化时间->Hz) 合成带谐波的旋律。"""
    rng = np.random.default_rng(seed)
    dur = 2.2 / speed
    t = np.linspace(0, dur, int(SR * dur), endpoint=False)
    tn = t / dur
    f0 = np.interp(tn, np.linspace(0, 1, len(contour)), contour)
    f0 = f0 * (1 + vib / 100.0 * np.sin(2 * np.pi * 5 * t))  # 轻微颤音
    phase = 2 * np.pi * np.cumsum(f0) / SR
    sig = (np.sin(phase)
           + 0.5 * np.sin(2 * phase)
           + 0.25 * np.sin(3 * phase))  # 三个谐波，制造频谱结构
    env = np.ones_like(t)
    att = int(0.05 * SR)
    env[:att] = np.linspace(0, 1, att)
    env[-att:] = np.linspace(1, 0, att)
    sig = sig * env
    if noise > 0:
        sig = sig + noise * rng.standard_normal(len(t))
    sig = sig / (np.max(np.abs(sig)) + 1e-9)
    return sig.astype(np.float32)


def save(path, y):
    sf.write(path, y, SR)


def main():
    # 同一段旋律的音高轮廓
    base = np.array([220, 262, 294, 330, 294, 262, 220, 196, 220, 262, 294, 330])
    other = np.array([330, 294, 262, 220, 196, 175, 196, 220, 262, 294, 330, 392])

    a = make_melody(base, seed=1)
    a_prime = make_melody(base, vib=6.0, noise=0.05, speed=1.08, seed=2)   # 模仿：抖动+噪+变速
    c = make_melody(other, seed=3)                                          # 无关旋律

    p_a, p_ap, p_c = "/tmp/vc_a.wav", "/tmp/vc_ap.wav", "/tmp/vc_c.wav"
    save(p_a, a); save(p_ap, a_prime); save(p_c, c)

    print("A   vs A    (同一段，期望最高):", compare(p_a, p_a))
    print("A   vs A'   (加噪模仿，期望较高):", compare(p_a, p_ap))
    print("A   vs C    (无关旋律，期望最低):", compare(p_a, p_c))

    for p in (p_a, p_ap, p_c):
        os.remove(p)


if __name__ == "__main__":
    main()
