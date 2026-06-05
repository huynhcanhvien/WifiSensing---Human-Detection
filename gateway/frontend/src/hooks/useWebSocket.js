import { useEffect, useRef, useCallback } from 'react';

const WS_URL = 'ws://localhost:8765';
const RECONNECT_MS = 3000;

export function useWebSocket({ onRawData, onAiResult, onLabelStatus, onStatusChange }) {
  const wsRef = useRef(null);
  const timerRef = useRef(null);

  // Keep handlers in a ref so the WebSocket closure always reads the latest version
  const handlers = useRef({ onRawData, onAiResult, onLabelStatus, onStatusChange });
  useEffect(() => {
    handlers.current = { onRawData, onAiResult, onLabelStatus, onStatusChange };
  });

  useEffect(() => {
    function connect() {
      handlers.current.onStatusChange('connecting');
      const ws = new WebSocket(WS_URL);
      wsRef.current = ws;

      ws.onopen = () => handlers.current.onStatusChange('connected');

      ws.onmessage = (evt) => {
        let p;
        try { p = JSON.parse(evt.data); } catch { return; }
        if (p.current_label !== undefined) handlers.current.onLabelStatus(p.current_label);
        if (p.type === 'raw_data') handlers.current.onRawData(p);
        else if (p.type === 'ai_result') handlers.current.onAiResult(p);
      };

      ws.onclose = () => {
        handlers.current.onStatusChange('disconnected');
        timerRef.current = setTimeout(connect, RECONNECT_MS);
      };

      ws.onerror = () => ws.close();
    }

    connect();
    return () => {
      if (timerRef.current) clearTimeout(timerRef.current);
      wsRef.current?.close();
    };
  }, []); // eslint-disable-line react-hooks/exhaustive-deps

  const send = useCallback((data) => {
    if (wsRef.current?.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(data));
    }
  }, []);

  return { send };
}
