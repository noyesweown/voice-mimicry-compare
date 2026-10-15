// embed.js — 浏览器端说话人嵌入（声纹）比对
// 依赖：ort.min.js（onnxruntime-web，全局 ort）
// 模型：Qwen3-Voice-Embedding ECAPA-TDNN，输出 1024 维声纹 x-vector
// 全程本地计算，不上传服务器。
(function (global) {
  'use strict';

  const SR = 24000, N_FFT = 1024, HOP = 256, N_MELS = 128, FMIN = 0, FMAX = 12000;
  const PI = Math.PI;

  function hz2mel(f) { return 1127 * Math.log(1 + f / 700); }
  function mel2hz(m) { return 700 * (Math.exp(m / 1127) - 1); }

  // librosa 风格 Slaney 归一化 mel 滤波器组
  function melFilterbank() {
    const nBins = N_FFT / 2 + 1;
    const fftFreqs = [];
    for (let i = 0; i < nBins; i++) fftFreqs.push(SR / 2 * i / (nBins - 1));
    const fMin = hz2mel(FMIN), fMax = hz2mel(FMAX);
    const mels = [];
    for (let i = 0; i < N_MELS + 2; i++) mels.push(fMin + (fMax - fMin) * i / (N_MELS + 1));
    const melHz = mels.map(mel2hz);
    const fdiff = [];
    for (let i = 0; i < nBins - 1; i++) fdiff.push(fftFreqs[i + 1] - fftFreqs[i]);
    const filters = [];
    for (let m = 0; m < N_MELS; m++) {
      const lo = melHz[m], c = melHz[m + 1], hi = melHz[m + 2];
      const w = new Float32Array(nBins);
      for (let k = 0; k < nBins; k++) {
        const f = fftFreqs[k];
        let v = 0;
        if (f >= lo && f <= c && c !== lo) v = (f - lo) / (c - lo);
        else if (f > c && f <= hi && hi !== c) v = (hi - f) / (hi - c);
        w[k] = Math.max(0, v);
      }
      for (let k = 0; k < nBins; k++) w[k] *= 2.0 / fdiff[Math.min(k, fdiff.length - 1)];
      filters.push(w);
    }
    return filters;
  }
  const MEL = melFilterbank();

  function hann(n) { const w = new Float32Array(n); for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * PI * i / (n - 1)); return w; }
  const HANN = hann(N_FFT);

  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) { let t = re[i]; re[i] = re[j]; re[j] = t; t = im[i]; im[i] = im[j]; im[j] = t; }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * PI / len; const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cwr = 1, cwi = 0; const half = len >> 1;
        for (let k = 0; k < half; k++) {
          const ur = re[i + k], ui = im[i + k];
          const vr = re[i + k + half] * cwr - im[i + k + half] * cwi;
          const vi = re[i + k + half] * cwi + im[i + k + half] * cwr;
          re[i + k] = ur + vr; im[i + k] = ui + vi;
          re[i + k + half] = ur - vr; im[i + k + half] = ui - vi;
          const ncwr = cwr * wr - cwi * wi; cwi = cwr * wi + cwi * wr; cwr = ncwr;
        }
      }
    }
  }

  function padReflect(x, pad) {
    const out = new Float32Array(x.length + 2 * pad);
    for (let i = 0; i < out.length; i++) {
      let idx = i - pad;
      if (idx < 0) idx = -idx;
      else if (idx >= x.length) idx = 2 * x.length - 2 - idx;
      out[i] = x[Math.max(0, Math.min(x.length - 1, idx))];
    }
    return out;
  }

  // 输入：单声道 Float32，采样率 SR；输出：(T,128) 的 mel 频谱（log 压缩）
  function melSpectrogram(sig) {
    const padded = padReflect(sig, N_FFT / 2);
    const frames = [];
    for (let s = 0; s + N_FFT <= padded.length; s += HOP) {
      const fr = new Float32Array(N_FFT);
      for (let i = 0; i < N_FFT; i++) fr[i] = padded[s + i] * HANN[i];
      const re = new Float32Array(N_FFT), im = new Float32Array(N_FFT);
      for (let i = 0; i < N_FFT; i++) re[i] = fr[i];
      fft(re, im);
      const nBins = N_FFT / 2 + 1;
      const mag = new Float32Array(nBins);
      for (let i = 0; i < nBins; i++) mag[i] = re[i] * re[i] + im[i] * im[i];
      frames.push(mag);
    }
    const T = frames.length;
    const out = new Float32Array(T * N_MELS);
    for (let t = 0; t < T; t++) {
      for (let m = 0; m < N_MELS; m++) {
        let s = 0;
        for (let k = 0; k < N_FFT / 2 + 1; k++) s += MEL[m][k] * frames[t][k];
        out[t * N_MELS + m] = Math.log(Math.max(s, 1e-5));
      }
    }
    return out; // length T*128
  }

  function cosine(a, b) {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    if (na === 0 || nb === 0) return 0;
    return d / (Math.sqrt(na) * Math.sqrt(nb));
  }

  let session = null;
  let loading = null;
  const MODEL_URL = './model/speaker_encoder_fp32.onnx';
  const MODEL_SIZE = 35943287; // 字节数，用于校验下载完整性

  // 初始化模型（首次调用时加载 wasm + onnx，约 62MB，之后走浏览器缓存）
  // 强制 wasm 后端：避免 Safari 等浏览器自动选择 WebNN/WebGPU 路径（对动态输入模型有兼容问题）
  function init() {
    if (session) return Promise.resolve(session);
    if (loading) return loading;
    if (typeof ort === 'undefined') return Promise.reject(new Error('onnxruntime-web 未加载（ort.min.js）'));
    ort.env.wasm.numThreads = 1;            // 单线程，免 COOP/COEP（GitHub Pages 友好）
    ort.env.wasm.wasmPaths = './';          // 与页面同目录加载 wasm
    loading = (async () => {
      // 先自行 fetch 并校验，再从字节建会话：能把「缓存了 404/被截断」这类问题变成明确报错
      const resp = await fetch(MODEL_URL);
      if (!resp.ok) throw new Error('模型下载失败 HTTP ' + resp.status + '，请刷新重试');
      const buf = await resp.arrayBuffer();
      if (buf.byteLength !== MODEL_SIZE) {
        throw new Error('模型文件不完整（收到 ' + buf.byteLength + '/' + MODEL_SIZE + ' 字节），请强制刷新（Cmd+Option+R）');
      }
      return ort.InferenceSession.create(buf, { executionProviders: ['wasm'] });
    })().then(s => { session = s; return s; }).catch(e => { loading = null; throw e; });
    return loading;
  }

  // 音频 -> 1024 维声纹向量
  async function embed(audio24k) {
    if (!session) await init();
    const mel = melSpectrogram(audio24k);
    const T = mel.length / N_MELS;
    const tensor = new ort.Tensor('float32', mel, [1, T, N_MELS]);
    const res = await session.run({ mel_spectrogram: tensor });
    return Float32Array.from(res.speaker_embedding.data);
  }

  // 比对两段 24k 单声道音频
  async function compare(orig, mimic) {
    const [e1, e2] = await Promise.all([embed(orig), embed(mimic)]);
    const c = cosine(e1, e2);
    return { cosine: c, score: Math.max(0, Math.min(1, c)) }; // score 0..1
  }

  // 任意采样率 AudioBuffer -> 24k 单声道 Float32
  function toMono(audioBuffer) {
    const ch = audioBuffer.getChannelData(0);
    const fromSr = audioBuffer.sampleRate, toSr = SR;
    const out = new Float32Array(Math.max(1, Math.floor(ch.length * toSr / fromSr)));
    if (fromSr === toSr) { out.set(ch.subarray(0, out.length)); }
    else {
      for (let i = 0; i < out.length; i++) {
        const x = i * (ch.length - 1) / (out.length - 1);
        const i0 = Math.floor(x), i1 = Math.min(i0 + 1, ch.length - 1);
        out[i] = ch[i0] + (ch[i1] - ch[i0]) * (x - i0);
      }
    }
    return out;
  }

  const api = { init, embed, compare, toMono, cosine, SR };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (global) global.Embed = api;
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
