// 用"不同音色"的合成信号验证算法能区分说话人/音色（正弦=原音，方波=模仿者）
const { compare } = require('./docs/core.js');

const SR = 16000;
function osc(type, freq, t) {
  const p = freq * t;
  if (type === 'square') return Math.sign(Math.sin(2 * Math.PI * p));
  if (type === 'saw') { const x = p - Math.floor(p + 0.5); return 2 * x; }
  return Math.sin(2 * Math.PI * p);
}
function tone(type, freq, dur, vibrato = 0, noise = 0) {
  const n = Math.floor(dur * SR);
  const out = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    const t = i / SR;
    const f = freq * (1 + vibrato * 0.01 * Math.sin(2 * Math.PI * 5 * t));
    out[i] = osc(type, f, t);
    if (noise) out[i] += noise * (Math.random() * 2 - 1);
  }
  return out;
}
function makeMelody(freqs, type, opts = {}) {
  const { vibrato = 0, noise = 0, speed = 1 } = opts;
  let x = new Float32Array(0);
  for (const f of freqs) {
    x = Float32Array.from([...x, ...tone(type, f, 0.3 / speed, vibrato, noise)]);
  }
  return x;
}

const base = [220, 262, 294, 330, 294, 262, 220, 196, 220, 262, 294, 330];
const other = [330, 294, 262, 220, 196, 175, 196, 220, 262, 294, 330, 392];

const A = makeMelody(base, 'sine');                  // 原音（说话人1）
const B = makeMelody(base, 'square');                // 模仿者（说话人2，同旋律不同音色）
const C = makeMelody(other, 'sine');                 // 不同旋律（同说话人1）

const rAA = compare(A, A);
const rAB = compare(A, B);
const rAC = compare(A, C);

const pct = v => (v * 100).toFixed(1) + '%';
console.log('A vs A   (同人同旋律) :', pct(rAA.total), '| mfcc', pct(rAA.mfcc), 'pitch', pct(rAB.pitch), 'energy', pct(rAA.energy));
console.log('A vs B   (同旋律不同人):', pct(rAB.total), '| mfcc', pct(rAB.mfcc), 'pitch', pct(rAB.pitch), 'energy', pct(rAB.energy));
console.log('A vs C   (不同旋律同人):', pct(rAC.total), '| mfcc', pct(rAC.mfcc), 'pitch', pct(rAC.pitch), 'energy', pct(rAC.energy));

const ok = rAA.total > 0.95 && rAB.total < 0.9 && isFinite(rAB.total) && isFinite(rAC.total);
console.log('\n算法能区分音色且排序合理?', ok ? 'YES ✅' : 'NO ❌');
process.exit(ok ? 0 : 1);
