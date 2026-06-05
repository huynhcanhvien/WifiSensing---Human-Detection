import { useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  CategoryScale, LinearScale,
  PointElement, LineElement, LineController,
  Title, Tooltip, Legend,
} from 'chart.js';
import { N_SC } from '../lib/dsp';

ChartJS.register(CategoryScale, LinearScale, PointElement, LineElement, LineController, Title, Tooltip, Legend);

const MAC0 = 'd4e9f4a36480';
const MAC1 = 'd4e9f4a43c18';
const GRID = '#374151';

const baseOpts = {
  responsive: true,
  maintainAspectRatio: false,
  animation: { duration: 0 },
  plugins: { legend: { display: false } },
};

export function LiveCharts({ csiBuffersRef }) {
  const waveRef = useRef(null);
  const specRef = useRef(null);
  const waveChart = useRef(null);
  const specChart = useRef(null);

  useEffect(() => {
    ChartJS.getChart(waveRef.current)?.destroy();
    ChartJS.getChart(specRef.current)?.destroy();

    waveChart.current = new ChartJS(waveRef.current, {
      type: 'line',
      data: {
        labels: [],
        datasets: [
          { data: [], borderColor: '#3B82F6', borderWidth: 1.2, pointRadius: 0, fill: false, tension: 0.2 },
          { data: [], borderColor: '#10B981', borderWidth: 1.2, pointRadius: 0, fill: false, tension: 0.2 },
        ],
      },
      options: { ...baseOpts, scales: { x: { display: false }, y: { grid: { color: GRID }, ticks: { color: '#6B7280', font: { size: 10 } } } } },
    });

    specChart.current = new ChartJS(specRef.current, {
      type: 'line',
      data: {
        labels: Array.from({ length: N_SC }, (_, i) => i),
        datasets: [
          { data: new Array(N_SC).fill(0), borderColor: '#3B82F6', borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.3 },
          { data: new Array(N_SC).fill(0), borderColor: '#10B981', borderWidth: 1.5, pointRadius: 0, fill: false, tension: 0.3 },
        ],
      },
      options: { ...baseOpts, scales: { x: { grid: { color: GRID }, ticks: { color: '#6B7280', font: { size: 10 } } }, y: { grid: { color: GRID }, ticks: { color: '#6B7280', font: { size: 10 } } } } },
    });

    return () => {
      waveChart.current?.destroy();
      waveChart.current = null;
      specChart.current?.destroy();
      specChart.current = null;
    };
  }, []);

  useEffect(() => {
    const id = setInterval(() => {
      const bufs = csiBuffersRef.current;
      [MAC0, MAC1].forEach((mac, mi) => {
        const buf = bufs[mac];
        if (buf.length < 5) return;
        const means = buf.map(f => f.sc.reduce((a, b) => a + b, 0) / N_SC);
        if (waveChart.current) {
          if (waveChart.current.data.labels.length !== means.length)
            waveChart.current.data.labels = new Array(means.length).fill('');
          waveChart.current.data.datasets[mi].data = means;
        }
        if (specChart.current)
          specChart.current.data.datasets[mi].data = Array.from(buf[buf.length - 1].sc);
      });
      waveChart.current?.update('none');
      specChart.current?.update('none');
    }, 50);
    return () => clearInterval(id);
  }, [csiBuffersRef]);

  return (
    <div className="grid grid-cols-2 gap-4 mb-4">
      <div className="bg-[#111827] border border-gray-700 rounded-lg p-4">
        <div className="text-xs font-medium text-gray-400 mb-2 flex items-center gap-2">
          Waveform — trung bình biên độ
          <span className="bg-gray-700 text-gray-300 px-2 py-0.5 rounded text-[11px]">Raw Hz · 10s</span>
        </div>
        <div style={{ position: 'relative', height: '130px' }}>
          <canvas ref={waveRef} />
        </div>
      </div>
      <div className="bg-[#111827] border border-gray-700 rounded-lg p-4">
        <div className="text-xs font-medium text-gray-400 mb-2">Phổ subcarrier — frame mới nhất</div>
        <div style={{ position: 'relative', height: '130px' }}>
          <canvas ref={specRef} />
        </div>
      </div>
    </div>
  );
}
