"""
一键录音: python record_clip.py 输出文件名.wav [秒数,默认5]
需要麦克风。录制后保存为 16kHz 单声道 wav，供 voice_compare_v2 定标使用。
"""
import sys
import sounddevice as sd
import soundfile as sf
import numpy as np

SR = 16000


def record(out_path, seconds=5):
    print(f"准备录音 {seconds} 秒，按下回车开始…")
    input()
    print("● 录音中… (Ctrl+C 取消)")
    audio = sd.rec(int(seconds * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    sf.write(out_path, audio, SR)
    print(f"✅ 已保存: {out_path}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法: python record_clip.py 输出文件名.wav [秒数]")
        sys.exit(1)
    out = sys.argv[1]
    secs = int(sys.argv[2]) if len(sys.argv) > 2 else 5
    record(out, secs)
