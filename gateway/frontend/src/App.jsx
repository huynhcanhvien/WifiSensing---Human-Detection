import { useState, useRef, useCallback } from 'react';
import { processSpatial, applyTemporalHampel } from './lib/dsp';
import { useWebSocket } from './hooks/useWebSocket';
import { Navbar }       from './components/Navbar';
import { Notification } from './components/Notification';
import { LivePage }      from './pages/LivePage';
import { HistoryPage }   from './pages/HistoryPage';
import { StatisticsPage } from './pages/StatisticsPage';

const MAC0    = 'd4e9f4a36480';
const MAC1    = 'd4e9f4a43c18';
const WIN_SEC = 10;

export default function App() {
  const [activeTab,    setActiveTab]    = useState('live');
  const [wsStatus,     setWsStatus]     = useState('connecting');
  const [aiResult,     setAiResult]     = useState({ action: '—', confidence: null, latency: null });
  const [currentLabel, setCurrentLabel] = useState(null);
  const [notifications, setNotifications] = useState([]);
  const [labels,       setLabels]       = useState(['Breathing', 'Empty']);

  const csiBuffersRef  = useRef({ [MAC0]: [], [MAC1]: [] });
  const notifIdRef     = useRef(0);
  const prevStatusRef  = useRef('connecting');
  const lastLowConfRef = useRef(0);

  const addNotification = useCallback((message, type = 'info') => {
    const id = ++notifIdRef.current;
    setNotifications(prev => [...prev, { id, message, type }]);
    setTimeout(() => setNotifications(prev => prev.filter(n => n.id !== id)), 4000);
  }, []);

  const handleRawData = useCallback((payload) => {
    const now = performance.now() / 1000;
    payload.data.forEach(f => {
      const mac = f.mac;
      if (mac !== MAC0 && mac !== MAC1) return;
      const spatial = processSpatial(f.amp);
      const buf = csiBuffersRef.current[mac];
      buf.push({ ts: now, spatial, sc: spatial.slice() });
      applyTemporalHampel(buf);
      while (buf.length > 1 && (now - buf[0].ts) > WIN_SEC) buf.shift();
    });
  }, []);

  const handleAiResult = useCallback((payload) => {
    setAiResult({
      action:     payload.action     ?? '—',
      confidence: payload.confidence ?? null,
      latency:    payload.latency_ms ?? null,
    });
    const conf = payload.confidence ?? 100;
    if (conf < 50) {
      const now = Date.now();
      if (now - lastLowConfRef.current > 10000) {
        addNotification(`Confidence thấp: ${conf.toFixed(1)}%`, 'warning');
        lastLowConfRef.current = now;
      }
    }
  }, [addNotification]);

  const handleLabelStatus = useCallback((label) => {
    setCurrentLabel(label || null);
  }, []);

  const handleStatusChange = useCallback((status) => {
    setWsStatus(status);
    const prev = prevStatusRef.current;
    if (status === 'disconnected' && prev === 'connected') {
      addNotification('Mất kết nối WebSocket. Đang thử lại...', 'error');
    } else if (status === 'connected' && prev === 'disconnected') {
      addNotification('Đã kết nối lại', 'success');
    }
    prevStatusRef.current = status;
  }, [addNotification]);

  const { send } = useWebSocket({
    onRawData:      handleRawData,
    onAiResult:     handleAiResult,
    onLabelStatus:  handleLabelStatus,
    onStatusChange: handleStatusChange,
  });

  return (
    <div className="min-h-screen bg-[#0b0f19] text-white">
      <Navbar activeTab={activeTab} onTabChange={setActiveTab} />
      <Notification notifications={notifications} />
      <main className="p-4 max-w-7xl mx-auto">
        {activeTab === 'live'       && (
          <LivePage
            wsStatus={wsStatus}
            aiResult={aiResult}
            currentLabel={currentLabel}
            csiBuffersRef={csiBuffersRef}
            labels={labels}
            setLabels={setLabels}
            send={send}
          />
        )}
        {activeTab === 'history'    && <HistoryPage />}
        {activeTab === 'statistics' && <StatisticsPage />}
      </main>
    </div>
  );
}
