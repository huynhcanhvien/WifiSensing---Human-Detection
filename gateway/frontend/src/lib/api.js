import { MOCK_STATS, queryInference } from './mockData';

const BASE      = '/api';
const PROBE_TTL = 30_000;   // re-check mỗi 30 s để tự phục hồi khi Pi kết nối lại

let _connected = null;      // null = chưa probe lần nào
let _probedAt  = 0;

/**
 * Probe một lần, cache kết quả PROBE_TTL ms.
 * Bất kỳ HTTP response nào = Pi online.
 * Network error / timeout (2.5 s) = Pi offline → dùng mock.
 */
async function isConnected() {
  if (_connected !== null && Date.now() - _probedAt < PROBE_TTL) return _connected;

  const ctrl = new AbortController();
  const tid  = setTimeout(() => ctrl.abort(), 2500);
  try {
    await fetch(`${BASE}/inference?limit=1`, { signal: ctrl.signal });
    clearTimeout(tid);
    _connected = true;
  } catch {
    clearTimeout(tid);
    _connected = false;
  }
  _probedAt = Date.now();
  return _connected;
}

async function get(path) {
  const res = await fetch(BASE + path);
  if (!res.ok) throw new Error(`API error ${res.status}: ${path}`);
  return res.json();
}

export async function fetchInference({ page = 1, limit = 50, action, from, to } = {}) {
  if (!(await isConnected())) return queryInference({ page, limit, action, from, to });

  const params = new URLSearchParams({ page, limit });
  if (action) params.set('action', action);
  if (from)   params.set('from', from);
  if (to)     params.set('to', to);
  return get(`/inference?${params}`);
}

export async function fetchInferenceStats() {
  if (!(await isConnected())) return MOCK_STATS;
  return get('/inference/stats');
}

export function exportInferenceCSV({ action, from, to } = {}) {
  // Dùng giá trị cache — đến lúc này Stats/History đã probe rồi
  if (_connected !== true) {
    const { data } = queryInference({ action, from, to, limit: 9999 });
    const header   = 'id,timestamp,predicted_action,confidence_score,latency_ms';
    const rows     = data.map(r =>
      `${r.id},${r.timestamp},${r.predicted_action},${r.confidence_score},${r.latency_ms}`
    );
    const blob = new Blob([[header, ...rows].join('\n')], { type: 'text/csv' });
    const url  = URL.createObjectURL(blob);
    const a    = Object.assign(document.createElement('a'), { href: url, download: 'inference_mock.csv' });
    a.click();
    URL.revokeObjectURL(url);
    return;
  }
  const params = new URLSearchParams();
  if (action) params.set('action', action);
  if (from)   params.set('from', from);
  if (to)     params.set('to', to);
  const qs = params.toString();
  window.open(BASE + '/inference/export' + (qs ? `?${qs}` : ''), '_blank');
}

export async function fetchLabels() {
  if (!(await isConnected())) return [];
  return get('/labels');
}

export async function fetchCSI({ page = 1, limit = 100, label, mac } = {}) {
  if (!(await isConnected())) return { data: [], total: 0, totalPages: 1 };
  const params = new URLSearchParams({ page, limit });
  if (label) params.set('label', label);
  if (mac)   params.set('mac', mac);
  return get(`/csi?${params}`);
}
