const STATUS_COLOR = { connected: '#10B981', disconnected: '#EF4444', connecting: '#F59E0B' };
const STATUS_TEXT  = { connected: 'Đã kết nối', disconnected: 'Mất kết nối', connecting: 'Đang kết nối...' };

export function StatusCards({ wsStatus, aiResult }) {
  const { action, confidence, latency } = aiResult;

  const confColor =
    confidence === null ? '#9CA3AF' :
    confidence >= 80    ? '#10B981' :
    confidence >= 50    ? '#F59E0B' : '#EF4444';

  return (
    <div className="grid grid-cols-4 gap-3 py-4">
      <Card label="Kết nối">
        <div className="flex items-center gap-2 text-sm font-medium">
          <span
            className={`w-2 h-2 rounded-full shrink-0 ${wsStatus === 'connected' ? 'animate-pulse' : ''}`}
            style={{ background: STATUS_COLOR[wsStatus] || '#F59E0B' }}
          />
          <span>{STATUS_TEXT[wsStatus] || 'Chưa xác định'}</span>
        </div>
      </Card>

      <Card label="Nhận dạng">
        <span className="text-xl font-semibold text-emerald-400">{action}</span>
      </Card>

      <Card label="Confidence">
        <span className="text-xl font-semibold" style={{ color: confColor }}>
          {confidence !== null ? `${confidence.toFixed(1)}%` : '—'}
        </span>
      </Card>

      <Card label="Latency AI">
        <span className="text-xl font-semibold text-white">
          {latency !== null ? `${latency.toFixed(0)} ms` : '—'}
        </span>
      </Card>
    </div>
  );
}

function Card({ label, children }) {
  return (
    <div className="bg-gray-800 rounded-lg px-4 py-3">
      <div className="text-xs text-gray-400 mb-1.5">{label}</div>
      {children}
    </div>
  );
}
