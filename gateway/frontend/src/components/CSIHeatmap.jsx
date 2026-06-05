import { useEffect, useRef } from 'react';
import { VIRIDIS_LUT, N_SC } from '../lib/dsp';

const WIN_SEC = 10;
const FLAT_SAMPLE = new Float32Array(50000);

export function CSIHeatmap({ mac, label, badge, csiBuffersRef }) {
  const canvasRef = useRef(null);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext('2d', { alpha: false });

    function render() {
      const buf = csiBuffersRef.current[mac];
      const nFrames = buf.length;
      if (nFrames === 0) return;

      const W = canvas.offsetWidth || 350;
      const H = 200;
      if (canvas.width !== W) canvas.width = W;
      if (canvas.height !== H) canvas.height = H;

      let ptr = 0;
      for (let i = 0; i < nFrames; i += 4) {
        const scArr = buf[i].sc;
        for (let j = 0; j < N_SC; j += 2) {
          if (scArr[j] > 0 && ptr < FLAT_SAMPLE.length) FLAT_SAMPLE[ptr++] = scArr[j];
        }
      }

      let vmax = 1;
      if (ptr > 0) {
        const active = FLAT_SAMPLE.subarray(0, ptr);
        active.sort();
        vmax = active[Math.floor(ptr * 0.95)];
      }

      const range = vmax || 1;
      const imgData = ctx.createImageData(W, H);
      const buf32 = new Uint32Array(imgData.data.buffer);

      for (let py = 0; py < H; py++) {
        let sc = Math.floor((H - 1 - py) * N_SC / H);
        if (sc < 0) sc = 0;
        if (sc >= N_SC) sc = N_SC - 1;

        for (let px = 0; px < W; px++) {
          let ti = Math.floor(px * nFrames / W);
          if (ti < 0) ti = 0;
          if (ti >= nFrames) ti = nFrames - 1;

          let t = buf[ti].sc[sc] / range;
          if (t < 0) t = 0;
          if (t > 1) t = 1;
          buf32[py * W + px] = VIRIDIS_LUT[Math.round(t * 255)];
        }
      }

      ctx.putImageData(imgData, 0, 0);

      ctx.fillStyle = 'rgba(255,255,255,0.65)';
      ctx.font = '10px Inter, sans-serif';
      ctx.textAlign = 'center';
      for (let s = 0; s <= WIN_SEC; s++) {
        const x = Math.round(s * (W / WIN_SEC));
        ctx.fillText(`${s}s`, Math.max(10, Math.min(x, W - 10)), H - 4);
      }
    }

    const id = setInterval(render, 50);
    return () => clearInterval(id);
  }, [mac, csiBuffersRef]);

  return (
    <div className="bg-[#111827] border border-gray-700 rounded-lg p-4">
      <div className="flex items-center gap-2 mb-2 text-xs font-medium text-gray-400">
        <span>{label}</span>
        <span className="bg-gray-700 text-gray-300 px-2 py-0.5 rounded text-[11px] font-mono">{badge}</span>
      </div>
      <canvas ref={canvasRef} style={{ width: '100%', height: '200px', display: 'block' }} />
    </div>
  );
}
