/**
 * mockData.js — Dữ liệu giả lập cho StatisticsPage và HistoryPage.
 *
 * Bật/tắt bằng cờ MOCK_MODE trong api.js.
 * Dữ liệu được sinh một lần khi module load (ổn định trong session).
 */

// ─── Helpers ──────────────────────────────────────────────────────────────────

function rand(min, max) {
  return Math.random() * (max - min) + min;
}

function randInt(min, max) {
  return Math.floor(rand(min, max + 1));
}

/** Trả về Date cách `now` một khoảng ngẫu nhiên trong [minMin, maxMin] phút. */
function randDate(now, minMin, maxMin) {
  return new Date(now.getTime() - rand(minMin, maxMin) * 60 * 1000);
}

function toTimestamp(d) {
  return d.toISOString().replace('T', ' ').slice(0, 19);
}

// ─── Config trạng thái ────────────────────────────────────────────────────────

const STATES = [
  {
    action:    'Breathing',
    weight:    0.55,
    confRange: [70, 98],
    latRange:  [110, 280],
  },
  {
    action:    'Empty',
    weight:    0.45,
    confRange: [78, 99],
    latRange:  [100, 260],
  },
];

// ─── Sinh bản ghi ─────────────────────────────────────────────────────────────

const TOTAL = 500;
const BASE_NOW = new Date();        // cố định cho cả session

function pickState() {
  const r = Math.random();
  let acc = 0;
  for (const s of STATES) {
    acc += s.weight;
    if (r < acc) return s;
  }
  return STATES[STATES.length - 1];
}

/** Tạo 500 bản ghi trải đều trên 8 giờ vừa rồi, sắp xếp mới nhất lên trước. */
const RAW_RECORDS = Array.from({ length: TOTAL }, () => {
  const state = pickState();
  const ts    = randDate(BASE_NOW, 0, 8 * 60);
  return {
    timestamp:        toTimestamp(ts),
    _ts:              ts,
    predicted_action: state.action,
    confidence_score: parseFloat(rand(...state.confRange).toFixed(1)),
    latency_ms:       parseFloat(rand(...state.latRange).toFixed(0)),
  };
}).sort((a, b) => b._ts - a._ts);

export const MOCK_RECORDS = RAW_RECORDS.map((r, i) => ({
  id:               TOTAL - i,
  timestamp:        r.timestamp,
  predicted_action: r.predicted_action,
  confidence_score: r.confidence_score,
  latency_ms:       r.latency_ms,
}));

// ─── fetchInference mock ───────────────────────────────────────────────────────

export function queryInference({ page = 1, limit = 50, action, from, to } = {}) {
  let rows = MOCK_RECORDS;

  if (action) {
    rows = rows.filter(r => r.predicted_action === action);
  }
  if (from) {
    const t = new Date(from.replace(' ', 'T'));
    rows = rows.filter(r => new Date(r.timestamp.replace(' ', 'T')) >= t);
  }
  if (to) {
    const t = new Date(to.replace(' ', 'T'));
    rows = rows.filter(r => new Date(r.timestamp.replace(' ', 'T')) <= t);
  }

  const total      = rows.length;
  const totalPages = Math.max(1, Math.ceil(total / limit));
  const safePage   = Math.min(page, totalPages);
  const data       = rows.slice((safePage - 1) * limit, safePage * limit);

  return { data, total, totalPages };
}

// ─── fetchInferenceStats mock ─────────────────────────────────────────────────

function buildStats() {
  const total = MOCK_RECORDS.length;

  // byAction
  const acc = {};
  for (const r of MOCK_RECORDS) {
    if (!acc[r.predicted_action]) acc[r.predicted_action] = { count: 0, confSum: 0, latSum: 0 };
    acc[r.predicted_action].count++;
    acc[r.predicted_action].confSum += r.confidence_score;
    acc[r.predicted_action].latSum  += r.latency_ms;
  }
  const byAction = Object.entries(acc).map(([action, s]) => ({
    predicted_action: action,
    count:            s.count,
    avg_confidence:   parseFloat((s.confSum / s.count).toFixed(1)),
    avg_latency:      parseFloat((s.latSum  / s.count).toFixed(0)),
  }));

  const avgConfidence = parseFloat(
    (MOCK_RECORDS.reduce((s, r) => s + r.confidence_score, 0) / total).toFixed(1)
  );
  const avgLatency = parseFloat(
    (MOCK_RECORDS.reduce((s, r) => s + r.latency_ms, 0) / total).toFixed(0)
  );

  // Latency histogram
  const BUCKETS = [
    { bucket: '< 150 ms',    test: l => l < 150 },
    { bucket: '150–200 ms',  test: l => l >= 150 && l < 200 },
    { bucket: '200–250 ms',  test: l => l >= 200 && l < 250 },
    { bucket: '250–300 ms',  test: l => l >= 250 && l < 300 },
    { bucket: '> 300 ms',    test: l => l >= 300 },
  ];
  const latencyHistogram = BUCKETS.map(b => ({
    bucket: b.bucket,
    count:  MOCK_RECORDS.filter(r => b.test(r.latency_ms)).length,
  }));

  // Confidence timeline — 60 phút gần nhất, mỗi bucket 1 phút
  const now = BASE_NOW;
  const confidenceTimeline = Array.from({ length: 60 }, (_, i) => {
    const minsAgo = 59 - i;
    const start   = new Date(now.getTime() - (minsAgo + 1) * 60 * 1000);
    const end     = new Date(now.getTime() - minsAgo * 60 * 1000);
    const bucket  = MOCK_RECORDS.filter(r => {
      const t = new Date(r.timestamp.replace(' ', 'T'));
      return t >= start && t < end;
    });
    const avg_conf = bucket.length > 0
      ? parseFloat((bucket.reduce((s, r) => s + r.confidence_score, 0) / bucket.length).toFixed(1))
      : 0;
    return { minute: start.toISOString().slice(0, 16), avg_conf };
  });

  return { total, avgConfidence, avgLatency, byAction, latencyHistogram, confidenceTimeline };
}

export const MOCK_STATS = buildStats();
