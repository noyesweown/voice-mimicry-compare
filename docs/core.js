// core.js — 纯前端语音相似度算法（整句级特征，v2）
// 浏览器：window.VoiceCompare.compare(originalFloat32, mimicFloat32) -> {mfcc,pitch,energy,total}
// Node  ：module.exports.compare(...)
(function (global) {
  'use strict';

  // ---- 参数（与 Python v2 对齐）----
  const SR = 16000;
  const WIN = 400;        // 25ms @16k
  const HOP = 160;        // 10ms @16k
  const N_FFT = 512;
  const N_MELS = 26;
  const N_MFCC = 13;
  const FMIN = 0, FMAX = 8000;
  const W_TIMBRE = 0.7, W_PITCH = 0.15, W_ENERGY = 0.15;
  const PITCH_CONF = 0.3; // 自相关置信阈值

  function linspace(a, b, n) {
    const out = new Array(n);
    for (let i = 0; i < n; i++) out[i] = a + (b - a) * i / (n - 1);
    return out;
  }

  // ---------- 迭代 FFT（Cooley-Tukey）----------
  function fft(re, im) {
    const n = re.length;
    for (let i = 1, j = 0; i < n; i++) {
      let bit = n >> 1;
      for (; j & bit; bit >>= 1) j ^= bit;
      j ^= bit;
      if (i < j) {
        let t = re[i]; re[i] = re[j]; re[j] = t;
        t = im[i]; im[i] = im[j]; im[j] = t;
      }
    }
    for (let len = 2; len <= n; len <<= 1) {
      const ang = -2 * Math.PI / len;
      const wr = Math.cos(ang), wi = Math.sin(ang);
      for (let i = 0; i < n; i += len) {
        let cwr = 1, cwi = 0;
        const half = len >> 1;
        for (let k = 0; k < half; k++) {
          const ur = re[i + k], ui = im[i + k];
          const vr = re[i + k + half] * cwr - im[i + k + half] * cwi;
          const vi = re[i + k + half] * cwi + im[i + k + half] * cwr;
          re[i + k] = ur + vr; im[i + k] = ui + vi;
          re[i + k + half] = ur - vr; im[i + k + half] = ui - vi;
          const ncwr = cwr * wr - cwi * wi;
          cwi = cwr * wi + cwi * wr; cwr = ncwr;
        }
      }
    }
  }

  // ---------- 梅尔滤波器组 ----------
  function melFilterbank() {
    const melMin = 1125 * Math.log(1 + FMIN / 700);
    const melMax = 1125 * Math.log(1 + FMAX / 700);
    const melPts = linspace(melMin, melMax, N_MELS + 2);
    const hzPts = melPts.map(m => 700 * (Math.exp(m / 1125) - 1));
    const binPts = hzPts.map(f => Math.floor((N_FFT + 1) * f / SR));
    const filters = [];
    for (let m = 1; m <= N_MELS; m++) {
      const fl = binPts[m - 1], fc = binPts[m], fr = binPts[m + 1];
      const filt = new Float32Array(N_FFT / 2 + 1);
      for (let k = fl; k <= fc && fc !== fl; k++) filt[k] = (k - fl) / (fc - fl);
      for (let k = fc; k < fr && fr !== fc; k++) filt[k] = (fr - k) / (fr - fc);
      filters.push(filt);
    }
    return filters;
  }
  const MEL_FB = melFilterbank();

  function hann(n) {
    const w = new Float32Array(n);
    for (let i = 0; i < n; i++) w[i] = 0.5 - 0.5 * Math.cos(2 * Math.PI * i / (n - 1));
    return w;
  }
  const HANN = hann(WIN);

  function frames(sig) {
    const out = [];
    for (let s = 0; s + WIN <= sig.length; s += HOP) {
      const fr = new Float32Array(WIN);
      for (let i = 0; i < WIN; i++) fr[i] = sig[s + i] * HANN[i];
      out.push(fr);
    }
    return out;
  }

  function frameMfcc(fr) {
    const re = new Float32Array(N_FFT), im = new Float32Array(N_FFT);
    for (let i = 0; i < WIN; i++) re[i] = fr[i];
    fft(re, im);
    const nBins = N_FFT / 2 + 1;
    const mag = new Float32Array(nBins);
    for (let i = 0; i < nBins; i++) mag[i] = re[i] * re[i] + im[i] * im[i];
    const logmel = new Float32Array(N_MELS);
    for (let m = 0; m < N_MELS; m++) {
      let s = 0;
      for (let k = 0; k < nBins; k++) s += mag[k] * MEL_FB[m][k];
      logmel[m] = Math.log(s + 1e-10);
    }
    const mfcc = new Float32Array(N_MFCC);
    for (let k = 0; k < N_MFCC; k++) {
      let sum = 0;
      for (let i = 0; i < N_MELS; i++) sum += logmel[i] * Math.cos(Math.PI * k * (2 * i + 1) / (2 * N_MELS));
      mfcc[k] = sum;
    }
    return mfcc;
  }

  // 把一组帧特征 -> 均值+标准差 向量（2D 维）
  function meanStd(mat) {
    const D = mat[0].length;
    const mean = new Float32Array(D), std = new Float32Array(D);
    for (let d = 0; d < D; d++) {
      let s = 0; for (let i = 0; i < mat.length; i++) s += mat[i][d];
      mean[d] = s / mat.length;
      let v = 0; for (let i = 0; i < mat.length; i++) { const x = mat[i][d] - mean[d]; v += x * x; }
      std[d] = Math.sqrt(v / mat.length) + 1e-9;
    }
    const vec = new Float32Array(D * 2);
    for (let d = 0; d < D; d++) { vec[d] = mean[d]; vec[D + d] = std[d]; }
    return vec;
  }

  function frameEnergy(fr) {
    let s = 0; for (let i = 0; i < fr.length; i++) s += fr[i] * fr[i];
    return Math.sqrt(s / fr.length + 1e-12);
  }

  function framePitch(fr) {
    const minLag = Math.floor(SR / 400), maxLag = Math.floor(SR / 70);
    let bestLag = -1, best = -Infinity;
    for (let lag = minLag; lag <= maxLag; lag++) {
      let s = 0;
      for (let i = 0; i + lag < WIN; i++) s += fr[i] * fr[i + lag];
      if (s > best) { best = s; bestLag = lag; }
    }
    let zero = 0; for (let i = 0; i < WIN; i++) zero += fr[i] * fr[i];
    const conf = best / (zero + 1e-9);
    if (conf < PITCH_CONF) return NaN;
    return SR / bestLag;
  }

  function interpolateNaN(arr) {
    const out = arr.slice();
    let prev = null;
    for (let i = 0; i < out.length; i++) {
      if (isNaN(out[i])) {
        let nxt = null;
        for (let j = i + 1; j < out.length; j++) { if (!isNaN(out[j])) { nxt = j; break; } }
        if (prev !== null && nxt !== null) out[i] = out[prev] + (out[nxt] - out[prev]) * (i - prev) / (nxt - prev);
        else if (prev !== null) out[i] = out[prev];
        else if (nxt !== null) out[i] = out[nxt];
        else out[i] = 0;
      }
      if (!isNaN(out[i])) prev = i;
    }
    return out;
  }

  function resample(arr, L) {
    if (arr.length === L) return arr.slice();
    const out = new Float32Array(L);
    for (let i = 0; i < L; i++) {
      const x = i * (arr.length - 1) / (L - 1);
      const i0 = Math.floor(x), i1 = Math.min(i0 + 1, arr.length - 1);
      out[i] = arr[i0] + (arr[i1] - arr[i0]) * (x - i0);
    }
    return out;
  }

  function zscore(arr) {
    let m = 0; for (let i = 0; i < arr.length; i++) m += arr[i];
    m /= arr.length;
    let sd = 0; for (let i = 0; i < arr.length; i++) { const x = arr[i] - m; sd += x * x; }
    sd = Math.sqrt(sd / arr.length) + 1e-9;
    return arr.map(v => (v - m) / sd);
  }

  // 余弦相似度，返回 [-1,1]
  function cosine(a, b) {
    let dot = 0, na = 0, nb = 0;
    for (let i = 0; i < a.length; i++) { dot += a[i] * b[i]; na += a[i] * a[i]; nb += b[i] * b[i]; }
    if (na === 0 || nb === 0) return 0;
    return dot / (Math.sqrt(na) * Math.sqrt(nb));
  }

  function to01(c) { return Math.max(0, Math.min(1, (c + 1) / 2)); }

  function extractFeatures(sig) {
    const fr = frames(sig);
    if (fr.length === 0) throw new Error('音频太短');
    const mfccs = fr.map(frameMfcc);
    const mfccVec = meanStd(mfccs);
    const pitch = fr.map(framePitch);
    const energy = fr.map(frameEnergy);
    const pitClean = interpolateNaN(pitch);
    const pitZ = zscore(pitClean.map(Math.log2));
    const enZ = zscore(energy);
    return {
      mfcc: mfccVec,
      pitch: resample(pitZ, 100),
      energy: resample(enZ, 100)
    };
  }

  function compare(a, b) {
    const fa = extractFeatures(a), fb = extractFeatures(b);
    const mfcc = to01(cosine(fa.mfcc, fb.mfcc));
    const pitch = to01(cosine(fa.pitch, fb.pitch));
    const energy = to01(cosine(fa.energy, fb.energy));
    const total = W_TIMBRE * mfcc + W_PITCH * pitch + W_ENERGY * energy;
    return { mfcc, pitch, energy, total };
  }

  // 浏览器/Node 自适应导出
  const api = { compare, extractFeatures, SR };
  if (typeof module !== 'undefined' && module.exports) module.exports = api;
  if (global) global.VoiceCompare = api;

})(typeof window !== 'undefined' ? window : (typeof globalThis !== 'undefined' ? globalThis : this));
