"""
record_and_compare.py
完整 demo：上传/指定原音频 → 录音 N 秒模仿 → 输出相似度。

依赖（在 voicecmp 环境里）：
  pip install sounddevice soundfile
运行：
  python record_and_compare.py 原音频.wav --seconds 5
然后对着麦克风模仿原音频，程序会录音并打分。
注意：麦克风录音依赖真实音频设备，请在本地（非沙箱）运行。
"""

import argparse
import numpy as np
import soundfile as sf
import sounddevice as sd
from voice_compare import compare, TARGET_SR


def record(seconds, sr=TARGET_SR):
    """用麦克风录指定秒数的单声道音频，返回 (波形, 采样率)。"""
    print(f"🎤 准备录音 {seconds} 秒，请模仿原音频……")
    audio = sd.rec(int(seconds * sr), samplerate=sr, channels=1, dtype="float32")
    sd.wait()
    print("✅ 录音结束")
    return audio[:, 0], sr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("original", help="原音频文件路径 (wav/mp3)")
    ap.add_argument("--seconds", type=float, default=5.0, help="录音时长(秒)")
    ap.add_argument("--out", default="/tmp/mimic_recorded.wav", help="录音保存路径")
    args = ap.parse_args()

    y, sr = record(args.seconds, TARGET_SR)
    sf.write(args.out, y, sr)

    res = compare(args.original, args.out)
    print("\n===== 模仿相似度结果 =====")
    print(f"音色相似度 : {res['mfcc_similarity']:.3f}")
    print(f"音高相似度 : {res['pitch_similarity']:.3f}")
    print(f"能量相似度 : {res['energy_similarity']:.3f}")
    print(f"综合相似度 : {res['total']:.3f}  ({res['total']*100:.1f}%)")


if __name__ == "__main__":
    main()
