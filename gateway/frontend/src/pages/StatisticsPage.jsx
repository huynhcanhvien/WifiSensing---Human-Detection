import { useState, useEffect, useRef } from 'react';
import {
  Chart as ChartJS,
  ArcElement, Tooltip, Legend,
  CategoryScale, LinearScale,
  PointElement, LineElement, LineController,
  BarElement, BarController,
  DoughnutController,
} from 'chart.js';
import { fetchInferenceStats } from '../lib/api';

ChartJS.register(
  ArcElement, Tooltip, Legend,
  CategoryScale, LinearScale,
  PointElement, LineElement, LineController,
  BarElement, BarController,
  DoughnutController,
);

const GRID  = '#374151';
const TICKS = { color: '#6B7280', font: { size: 10 } };

function StatCard({ label, value, sub, color }) {
  return (
    <div className="bg-[#111827] border border-gray-700 rounded-lg px-5 py-4">
      <div className="text-xs text-gray-400 mb-1">{label}</div>
      <div className="text-2xl font-bold" style={{ color: color || '#fff' }}>{value}</div>
      {sub && <div className="text-xs text-gray-500 mt-0.5">{sub}</div>}
    </div>
  );
}

function ChartPanel({ title, badge, children }) {
  return (
    <div className="bg-[#111827] border border-gray-700 rounded-lg p-4">
      <div className="text-xs font-medium text-gray-400 mb-3 flex items-center gap-2">
        {title}
        {badge && <span className="bg-gray-700 text-gray-300 px-2 py-0.5 rounded text-[11px]">{badge}</span>}
      </div>
      {children}
    </div>
  );
}

function DoughnutChart({ data }) {
  const ref   = useRef(null);
  const chart = useRef(null);

  useEffect(() => {
    ChartJS.getChart(ref.current)?.destroy();
    if (!data?.byAction?.length) return;

    chart.current = new ChartJS(ref.current, {
      type: 'doughnut',
      data: {
        labels:   data.byAction.map(a => a.predicted_action),
        datasets: [{
          data:            data.byAction.map(a => a.count),
          backgroundColor: ['#3B82F6', '#10B981', '#F59E0B', '#EF4444', '#8B5CF6'],
          borderWidth:     0,
          hoverOffset:     6,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 600 },
        plugins: {
          legend: { position: 'right', labels: { color: '#9CA3AF', font: { size: 11 }, padding: 14 } },
          tooltip: {
            callbacks: {
              label: ctx => ` ${ctx.label}: ${ctx.parsed} lần (${((ctx.parsed / data.total) * 100).toFixed(1)}%)`,
            },
          },
        },
      },
    });
    return () => { chart.current?.destroy(); chart.current = null; };
  }, [data]);

  return <div style={{ height: '200px' }}><canvas ref={ref} /></div>;
}

function TimelineChart({ data }) {
  const ref   = useRef(null);
  const chart = useRef(null);

  useEffect(() => {
    ChartJS.getChart(ref.current)?.destroy();
    if (!data?.length) return;

    chart.current = new ChartJS(ref.current, {
      type: 'line',
      data: {
        labels:   data.map(d => d.minute.slice(11)),
        datasets: [{
          label:       'Avg Confidence',
          data:        data.map(d => d.avg_conf),
          borderColor: '#3B82F6',
          borderWidth: 1.5,
          pointRadius: 0,
          fill:        {
            target: 'origin',
            above:  'rgba(59,130,246,0.08)',
          },
          tension: 0.3,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { color: GRID }, ticks: { ...TICKS, maxTicksLimit: 8 } },
          y: { grid: { color: GRID }, ticks: TICKS, min: 0, max: 100,
               title: { display: true, text: '%', color: '#6B7280', font: { size: 10 } } },
        },
      },
    });
    return () => { chart.current?.destroy(); chart.current = null; };
  }, [data]);

  return <div style={{ height: '180px' }}><canvas ref={ref} /></div>;
}

function LatencyChart({ data }) {
  const ref   = useRef(null);
  const chart = useRef(null);

  useEffect(() => {
    ChartJS.getChart(ref.current)?.destroy();
    if (!data?.length) return;

    chart.current = new ChartJS(ref.current, {
      type: 'bar',
      data: {
        labels:   data.map(d => d.bucket),
        datasets: [{
          data:            data.map(d => d.count),
          backgroundColor: data.map((_, i) =>
            ['#10B981', '#3B82F6', '#F59E0B', '#F97316', '#EF4444'][i] || '#6B7280'
          ),
          borderRadius: 4,
          borderWidth:  0,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        animation: { duration: 400 },
        plugins: { legend: { display: false } },
        scales: {
          x: { grid: { display: false }, ticks: TICKS },
          y: { grid: { color: GRID }, ticks: TICKS,
               title: { display: true, text: 'Số lần', color: '#6B7280', font: { size: 10 } } },
        },
      },
    });
    return () => { chart.current?.destroy(); chart.current = null; };
  }, [data]);

  return <div style={{ height: '180px' }}><canvas ref={ref} /></div>;
}

export function StatisticsPage() {
  const [stats,   setStats]   = useState(null);
  const [loading, setLoading] = useState(true);
  const [error,   setError]   = useState(null);

  useEffect(() => {
    fetchInferenceStats()
      .then(setStats)
      .catch(() => setError('Không thể tải thống kê. Kiểm tra API server đang chạy (port 8000).'))
      .finally(() => setLoading(false));
  }, []);

  if (loading) {
    return (
      <div className="flex items-center justify-center h-64 text-gray-500">Đang tải thống kê...</div>
    );
  }

  if (error) {
    return (
      <div className="py-4">
        <div className="bg-red-950 border border-red-700 text-red-300 rounded-lg px-4 py-3 text-sm">{error}</div>
      </div>
    );
  }

  if (!stats || stats.total === 0) {
    return (
      <div className="flex flex-col items-center justify-center h-64 gap-3">
        <div className="text-5xl">📊</div>
        <div className="text-white font-medium">Thống Kê</div>
        <div className="text-gray-500 text-sm">Chưa có dữ liệu inference. Chạy hệ thống để thu thập dữ liệu.</div>
      </div>
    );
  }

  const confColor = stats.avgConfidence >= 80 ? '#10B981' : stats.avgConfidence >= 50 ? '#F59E0B' : '#EF4444';
  const latColor  = stats.avgLatency < 200 ? '#10B981' : stats.avgLatency < 400 ? '#F59E0B' : '#EF4444';

  return (
    <div className="py-4 space-y-5">
      <div>
        <h2 className="text-lg font-semibold text-white">Thống Kê Hệ Thống</h2>
        <p className="text-gray-500 text-sm mt-0.5">Phân tích {stats.total.toLocaleString()} lần nhận dạng</p>
      </div>

      {/* Summary cards */}
      <div className="grid grid-cols-4 gap-3">
        <StatCard label="Tổng inferences"  value={stats.total.toLocaleString()} sub="toàn bộ thời gian" />
        <StatCard label="Avg Confidence"   value={`${stats.avgConfidence}%`}    sub="trung bình"        color={confColor} />
        <StatCard label="Avg Latency"      value={`${stats.avgLatency} ms`}     sub="trung bình"        color={latColor} />
        <StatCard label="Số actions"       value={stats.byAction.length}        sub="loại nhận dạng" />
      </div>

      {/* Action breakdown table */}
      <div className="grid grid-cols-2 gap-4">
        <ChartPanel title="Phân bố Action" badge={`${stats.total} inferences`}>
          <DoughnutChart data={stats} />
          <div className="mt-3 space-y-1.5">
            {stats.byAction.map(a => (
              <div key={a.predicted_action} className="flex items-center justify-between text-xs">
                <span className="text-gray-300">{a.predicted_action}</span>
                <div className="flex gap-4 text-gray-400">
                  <span>{a.count} lần</span>
                  <span className="text-emerald-400">{a.avg_confidence}%</span>
                  <span className="text-blue-400">{a.avg_latency} ms</span>
                </div>
              </div>
            ))}
          </div>
        </ChartPanel>

        <ChartPanel title="Latency Distribution" badge="phân phối">
          <LatencyChart data={stats.latencyHistogram} />
        </ChartPanel>
      </div>

      {/* Confidence timeline */}
      <ChartPanel title="Confidence theo thời gian" badge="avg/phút · 60 phút gần nhất">
        <TimelineChart data={stats.confidenceTimeline} />
      </ChartPanel>
    </div>
  );
}
