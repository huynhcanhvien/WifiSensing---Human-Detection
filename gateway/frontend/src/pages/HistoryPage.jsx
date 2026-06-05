import { useState, useEffect, useCallback } from 'react';
import { fetchInference, exportInferenceCSV } from '../lib/api';

const CONF_COLOR = (c) =>
  c >= 80 ? '#10B981' : c >= 50 ? '#F59E0B' : '#EF4444';

const ACTIONS = ['', 'Breathing', 'Empty'];

function FilterBar({ filters, onChange, onExport, loading }) {
  return (
    <div className="flex flex-wrap gap-3 items-end mb-5">
      <div>
        <label className="text-xs text-gray-400 block mb-1">Action</label>
        <select
          value={filters.action}
          onChange={e => onChange({ ...filters, action: e.target.value, page: 1 })}
          className="bg-[#111827] border border-gray-700 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
        >
          {ACTIONS.map(a => (
            <option key={a} value={a}>{a || 'Tất cả'}</option>
          ))}
        </select>
      </div>
      <div>
        <label className="text-xs text-gray-400 block mb-1">Từ ngày</label>
        <input
          type="datetime-local"
          value={filters.from}
          onChange={e => onChange({ ...filters, from: e.target.value, page: 1 })}
          className="bg-[#111827] border border-gray-700 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
        />
      </div>
      <div>
        <label className="text-xs text-gray-400 block mb-1">Đến ngày</label>
        <input
          type="datetime-local"
          value={filters.to}
          onChange={e => onChange({ ...filters, to: e.target.value, page: 1 })}
          className="bg-[#111827] border border-gray-700 rounded-md px-3 py-2 text-sm text-white focus:outline-none focus:border-blue-500"
        />
      </div>
      <button
        onClick={() => onChange({ action: '', from: '', to: '', page: 1 })}
        className="px-4 py-2 rounded-md text-sm bg-gray-700 hover:bg-gray-600 text-gray-300 transition-colors"
      >
        Xóa lọc
      </button>
      <button
        onClick={onExport}
        disabled={loading}
        className="ml-auto px-4 py-2 rounded-md text-sm bg-emerald-700 hover:bg-emerald-600 text-white font-medium transition-colors disabled:opacity-50"
      >
        ↓ Tải CSV
      </button>
    </div>
  );
}

function Pagination({ page, totalPages, onChange }) {
  if (totalPages <= 1) return null;
  const pages = [];
  const start = Math.max(1, page - 2);
  const end   = Math.min(totalPages, page + 2);
  for (let i = start; i <= end; i++) pages.push(i);

  return (
    <div className="flex items-center justify-center gap-1 mt-5">
      <PageBtn disabled={page === 1}        onClick={() => onChange(page - 1)}>‹</PageBtn>
      {start > 1 && <><PageBtn onClick={() => onChange(1)}>1</PageBtn><span className="text-gray-600 px-1">…</span></>}
      {pages.map(p => (
        <PageBtn key={p} active={p === page} onClick={() => onChange(p)}>{p}</PageBtn>
      ))}
      {end < totalPages && <><span className="text-gray-600 px-1">…</span><PageBtn onClick={() => onChange(totalPages)}>{totalPages}</PageBtn></>}
      <PageBtn disabled={page === totalPages} onClick={() => onChange(page + 1)}>›</PageBtn>
    </div>
  );
}

function PageBtn({ children, active, disabled, onClick }) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className={`w-8 h-8 rounded text-sm font-medium transition-colors ${
        active    ? 'bg-blue-600 text-white' :
        disabled  ? 'text-gray-700 cursor-not-allowed' :
                    'text-gray-400 hover:bg-gray-700 hover:text-white'
      }`}
    >
      {children}
    </button>
  );
}

export function HistoryPage() {
  const [data,       setData]       = useState([]);
  const [total,      setTotal]      = useState(0);
  const [totalPages, setTotalPages] = useState(1);
  const [loading,    setLoading]    = useState(false);
  const [error,      setError]      = useState(null);
  const [filters,    setFilters]    = useState({ action: '', from: '', to: '', page: 1 });

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const { action, from, to, page } = filters;
      const res = await fetchInference({
        page,
        limit: 50,
        action: action || undefined,
        from:   from   ? from.replace('T', ' ') : undefined,
        to:     to     ? to.replace('T', ' ')   : undefined,
      });
      setData(res.data);
      setTotal(res.total);
      setTotalPages(res.totalPages);
    } catch (e) {
      setError('Không thể tải dữ liệu. Kiểm tra API server đang chạy (port 8000).');
    } finally {
      setLoading(false);
    }
  }, [filters]);

  useEffect(() => { load(); }, [load]);

  const handleExport = () => {
    const { action, from, to } = filters;
    exportInferenceCSV({
      action: action || undefined,
      from:   from   ? from.replace('T', ' ') : undefined,
      to:     to     ? to.replace('T', ' ')   : undefined,
    });
  };

  return (
    <div className="py-4">
      <div className="mb-4">
        <h2 className="text-lg font-semibold text-white">Lịch Sử Nhận Dạng</h2>
        <p className="text-gray-500 text-sm mt-0.5">
          {total > 0 ? `${total.toLocaleString()} kết quả` : 'Chưa có dữ liệu'}
        </p>
      </div>

      <FilterBar filters={filters} onChange={setFilters} onExport={handleExport} loading={loading} />

      {error && (
        <div className="bg-red-950 border border-red-700 text-red-300 rounded-lg px-4 py-3 text-sm mb-4">
          {error}
        </div>
      )}

      <div className="bg-[#111827] border border-gray-700 rounded-lg overflow-hidden">
        <table className="w-full text-sm">
          <thead>
            <tr className="border-b border-gray-700">
              <th className="text-left px-4 py-3 text-xs font-medium text-gray-400 w-16">#</th>
              <th className="text-left px-4 py-3 text-xs font-medium text-gray-400">Thời gian</th>
              <th className="text-left px-4 py-3 text-xs font-medium text-gray-400">Action</th>
              <th className="text-right px-4 py-3 text-xs font-medium text-gray-400">Confidence</th>
              <th className="text-right px-4 py-3 text-xs font-medium text-gray-400">Latency</th>
            </tr>
          </thead>
          <tbody>
            {loading && (
              <tr>
                <td colSpan={5} className="text-center py-12 text-gray-500">Đang tải...</td>
              </tr>
            )}
            {!loading && data.length === 0 && (
              <tr>
                <td colSpan={5} className="text-center py-12 text-gray-500">Không có dữ liệu</td>
              </tr>
            )}
            {!loading && data.map((row, i) => (
              <tr
                key={row.id}
                className="border-t border-gray-800 hover:bg-gray-800/40 transition-colors"
              >
                <td className="px-4 py-2.5 text-gray-600 font-mono text-xs">{row.id}</td>
                <td className="px-4 py-2.5 text-gray-300 font-mono text-xs">{row.timestamp}</td>
                <td className="px-4 py-2.5">
                  <span className={`inline-block px-2.5 py-0.5 rounded-full text-xs font-medium ${
                    row.predicted_action === 'Breathing'
                      ? 'bg-blue-900/50 text-blue-300'
                      : 'bg-gray-700 text-gray-300'
                  }`}>
                    {row.predicted_action}
                  </span>
                </td>
                <td className="px-4 py-2.5 text-right font-mono font-semibold" style={{ color: CONF_COLOR(row.confidence_score) }}>
                  {row.confidence_score?.toFixed(1)}%
                </td>
                <td className="px-4 py-2.5 text-right text-gray-400 font-mono text-xs">
                  {row.latency_ms?.toFixed(0)} ms
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <Pagination page={filters.page} totalPages={totalPages} onChange={p => setFilters(f => ({ ...f, page: p }))} />
    </div>
  );
}
