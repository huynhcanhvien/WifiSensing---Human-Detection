export const N_SC = 64;

const NULL_SET = new Set([0,1,2,3,4,5,29,30,31,32,33,34,59,60,61,62,63]);
const VALID = Array.from({ length: 64 }, (_, i) => i).filter(i => !NULL_SET.has(i));

export function processSpatial(amp64) {
  const shift = new Float32Array(64);
  for (let i = 0; i < 32; i++) { shift[i] = amp64[i + 32]; shift[i + 32] = amp64[i]; }

  const out = new Float32Array(64);
  for (let i = 0; i < 64; i++) {
    if (NULL_SET.has(i)) {
      let l = -1, r = -1;
      for (const v of VALID) { if (v < i) l = v; if (v > i && r === -1) r = v; }
      if (l !== -1 && r !== -1) out[i] = shift[l] + ((i - l) / (r - l)) * (shift[r] - shift[l]);
      else if (l !== -1) out[i] = shift[l];
      else if (r !== -1) out[i] = shift[r];
    } else {
      out[i] = shift[i];
    }
  }
  return out;
}

export function applyTemporalHampel(buf) {
  const L = buf.length;
  if (L < 5) return;
  const idx = L - 3;
  const filtered = new Float32Array(N_SC);
  for (let sc = 0; sc < N_SC; sc++) {
    const w = [
      buf[idx - 2].spatial[sc], buf[idx - 1].spatial[sc], buf[idx].spatial[sc],
      buf[idx + 1].spatial[sc], buf[idx + 2].spatial[sc],
    ].sort((a, b) => a - b);
    const med = w[2];
    const mads = [
      Math.abs(buf[idx - 2].spatial[sc] - med), Math.abs(buf[idx - 1].spatial[sc] - med),
      Math.abs(buf[idx].spatial[sc] - med), Math.abs(buf[idx + 1].spatial[sc] - med),
      Math.abs(buf[idx + 2].spatial[sc] - med),
    ].sort((a, b) => a - b);
    const mad = mads[2] * 1.4826;
    filtered[sc] = Math.abs(buf[idx].spatial[sc] - med) > 3 * mad ? med : buf[idx].spatial[sc];
  }
  buf[idx].sc = filtered;
}

export const VIRIDIS_LUT = (() => {
  const lut = new Uint32Array(256);
  const stops = [
    [0, 68, 1, 84], [64, 59, 82, 139], [128, 33, 145, 140], [192, 94, 201, 98], [255, 253, 231, 37],
  ];
  for (let i = 0; i < 256; i++) {
    let lo = 0;
    for (let s = 0; s < stops.length - 1; s++) if (stops[s][0] <= i) lo = s;
    const hi = Math.min(lo + 1, stops.length - 1);
    const a = stops[lo], b = stops[hi];
    const span = b[0] - a[0] || 1;
    const frac = (i - a[0]) / span;
    const r  = Math.round(a[1] + (b[1] - a[1]) * frac);
    const g  = Math.round(a[2] + (b[2] - a[2]) * frac);
    const bl = Math.round(a[3] + (b[3] - a[3]) * frac);
    lut[i] = (255 << 24) | (bl << 16) | (g << 8) | r;
  }
  return lut;
})();
