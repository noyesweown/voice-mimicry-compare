// embed.js — 浏览器端「声纹相似度」（SpeechBrain ECAPA-TDNN，VoxCeleb 训练）
// 依赖：ort.min.js（onnxruntime-web 1.19.2，全局 ort）
// 模型：ecapa_int8.onnx（192 维声纹嵌入，int8 量化，21.5MB）
// 预处理严格对齐官方 SpeechBrain：功率谱 -> fbank -> 10*log10 -> 80dB 截断 -> 逐句减均值
// 全程本地计算，不上传服务器。
(function (global) {
  'use strict';

  const SR = 16000, N_FFT = 400, WIN = 400, HOP = 160, N_BINS = 201, N_MELS = 80;
  const MODEL_URL = './model/ecapa_fp32.onnx';
  const FBANK_URL = './fbank_201x80.bin';   // 官方滤波器矩阵 (201,80) float32

  let W = null;            // Float32Array(201*80)
  let session = null;
  let loading = null;

  // ---- 预计算 DFT 核（N_FFT=400 不是 2 的幂，直接用矩阵法算功率谱）----
  let COS = null, SIN = null;
  function buildDFT() {
    COS = new Float32Array(N_BINS * N_FFT);
    SIN = new Float32Array(N_BINS * N_FFT);
    for (let k = 0; k < N_BINS; k++) {
      for (let n = 0; n < N_FFT; n++) {
        const a = 2 * Math.PI * k * n / N_FFT;
        COS[k * N_FFT + n] = Math.cos(a);
        SIN[k * N_FFT + n] = Math.sin(a);
      }
    }
  }
  buildDFT();

  // torch 风格 hann(periodic)
  const HANN = (function () {
    const w = new Float32Array(WIN);
    for (let i = 0; i < WIN; i++) w[i] = 0.5 * (1 - Math.cos(2 * Math.PI * i / WIN));
    return w;
  })();

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

  // 音频(16k 单声道) -> fbank 特征 (T,80)，与官方 SpeechBrain 一致
  function computeFeatures(sig) {
    const padded = padReflect(sig, N_FFT / 2);
    const T = Math.floor((padded.length - WIN) / HOP) + 1;
    if (T < 1) throw new Error('音频太短');
    const lin = new Float32Array(T * N_MELS);
    const pow = new Float32Array(N_BINS);

    for (let t = 0; t < T; t++) {
      const off = t * HOP;
      // 1) 功率谱
      for (let k = 0; k < N_BINS; k++) {
        let re = 0, im = 0;
        const base = k * N_FFT;
        for (let n = 0; n < N_FFT; n++) {
          const v = padded[off + n] * HANN[n];
          re += v * COS[base + n];
          im -= v * SIN[base + n];
        }
        pow[k] = re * re + im * im;
      }
      // 2) matmul(pow, W) -> (80,)
      for (let m = 0; m < N_MELS; m++) {
        let s = 0;
        for (let k = 0; k < N_BINS; k++) s += pow[k] * W[k * N_MELS + m];
        lin[t * N_MELS + m] = s;
      }
    }
    // 3) 10*log10，找全局最大值用于 80dB 截断
    let gmax = -Infinity;
    for (let i = 0; i < lin.length; i++) {
      const db = 10 * Math.log10(Math.max(lin[i], 1e-10));
      lin[i] = db;
      if (db > gmax) gmax = db;
    }
    // 4) 动态范围截断 + 逐句逐带减均值
    const floor = gmax - 80;
    const out = new Float32Array(T * N_MELS);
    const mean = new Float32Array(N_MELS);
    for (let t = 0; t < T; t++) {
      for (let m = 0; m < N_MELS; m++) {
        const v = Math.max(lin[t * N_MELS + m], floor);
        out[t * N_MELS + m] = v;
        mean[m] += v / T;
      }
    }
    for (let t = 0; t < T; t++) {
      for (let m = 0; m < N_MELS; m++) out[t * N_MELS + m] -= mean[m];
    }
    return out;
  }

  function cosine(a, b) {
    let d = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { d += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    if (na === 0 || nb === 0) return 0;
    return d / (Math.sqrt(na) * Math.sqrt(nb));
  }

  async function init() {
    if (session) return session;
    if (loading) return loading;
    if (typeof ort === 'undefined') throw new Error('onnxruntime-web 未加载（ort.min.js）');
    ort.env.wasm.numThreads = 1;     // 单线程，免 COOP/COEP（GitHub Pages 友好）
    ort.env.wasm.wasmPaths = './';   // wasm 与页面同目录
    loading = (async () => {
      if (!W) {
        const r = await fetch(FBANK_URL);
        if (!r.ok) throw new Error('滤波器矩阵加载失败 HTTP ' + r.status);
        const b = await r.arrayBuffer();
        W = new Float32Array(b);
      }
      return ort.InferenceSession.create(MODEL_URL, { executionProviders: ['wasm'] });
    })().then(s => { session = s; return s; }).catch(e => { loading = null; throw e; });
    return loading;
  }

  async function embed(audio16k) {
    if (!session) await init();
    const f = computeFeatures(audio16k);
    const T = f.length / N_MELS;
    const ft = new ort.Tensor('float32', f, [1, T, N_MELS]);
    const lens = new ort.Tensor('float32', Float32Array.from([1.0]), [1]);
    const res = await session.run({ features: ft, feature_lens: lens });
    return res.embedding.data;
  }

  // 比对：返回 { cosine, score }（score 为 0..1 的展示分，直接取余弦并裁剪）
  async function compare(orig, mimic) {
    const [a, b] = await Promise.all([embed(orig), embed(mimic)]);
    const c = cosine(a, b);
    return { cosine: c, score: Math.max(0, Math.min(1, c)) };
  }

  // AudioBuffer -> 16k 单声道 Float32
  function toMono(audioBuffer) {
    const ch = audioBuffer.getChannelData(0);
    const fromSr = audioBuffer.sampleRate;
    if (fromSr === SR) return Float32Array.from(ch.subarray(0, ch.length));
    const out = new Float32Array(Math.max(1, Math.floor(ch.length * SR / fromSr)));
    for (let i = 0; i < out.length; i++) {
      const x = i * (ch.length - 1) / (out.length - 1);
      const i0 = Math.floor(x), i1 = Math.min(i0 + 1, ch.length - 1);
      out[i] = ch[i0] + (ch[i1] - ch[i0]) * (x - i0);
    }
    return out;
  }

  const api = { init, embed, compare, toMono, cosine, computeFeatures: computeFeatures, SR };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (global) global.Embed = api;
})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
