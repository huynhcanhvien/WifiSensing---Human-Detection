import { useState } from 'react';

const DEFAULT_LABELS = ['Breathing', 'Empty'];

export function LabelingPanel({ labels, setLabels, currentLabel, send }) {
  const [selected, setSelected] = useState(labels[0]);
  const [sessionName, setSessionName] = useState('');
  const [newInput, setNewInput] = useState('');

  const isRecording = !!currentLabel;

  const handleStart = () => send({ cmd: 'start_label', label: selected });
  const handleStop  = () => send({ cmd: 'stop_label' });

  const addLabel = () => {
    const t = newInput.trim();
    if (t && !labels.includes(t)) {
      setLabels(prev => [...prev, t]);
      setSelected(t);
    }
    setNewInput('');
  };

  const removeLabel = (lbl) => {
    const next = labels.filter(l => l !== lbl);
    setLabels(next);
    if (selected === lbl) setSelected(next[0] ?? '');
  };

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg p-6 mt-4">
      <div className="text-base font-semibold text-white mb-0.5">Gắn Nhãn Dữ Liệu</div>
      <div className="text-sm text-gray-400 mb-5">Chọn nhãn và bắt đầu ghi để lưu vào Database.</div>

      {/* Session name */}
      <div className="mb-4">
        <label className="text-xs text-gray-400 block mb-1.5">Tên phiên (tuỳ chọn)</label>
        <input
          type="text"
          value={sessionName}
          onChange={e => setSessionName(e.target.value)}
          placeholder="Ví dụ: Thử nghiệm 1"
          disabled={isRecording}
          className="bg-[#111827] border border-gray-700 rounded-md px-3 py-2 text-sm text-white placeholder-gray-600 w-full max-w-xs focus:outline-none focus:border-blue-500 disabled:opacity-50 transition-colors"
        />
      </div>

      {/* Label selector */}
      <div className="mb-5">
        <label className="text-xs text-gray-400 block mb-2">Chọn nhãn</label>
        <div className="flex flex-wrap gap-2 items-center">
          {labels.map(lbl => (
            <div key={lbl} className="flex items-center gap-0.5">
              <button
                onClick={() => setSelected(lbl)}
                disabled={isRecording}
                className={`px-4 py-2 rounded-md text-sm font-medium border transition-colors disabled:opacity-60 ${
                  selected === lbl
                    ? 'bg-blue-600 border-blue-500 text-white'
                    : 'bg-[#111827] border-gray-700 text-gray-300 hover:border-gray-500'
                }`}
              >
                {lbl}
              </button>
              {!DEFAULT_LABELS.includes(lbl) && !isRecording && (
                <button
                  onClick={() => removeLabel(lbl)}
                  className="text-gray-600 hover:text-red-400 text-xs w-5 h-5 flex items-center justify-center rounded transition-colors"
                  title="Xoá nhãn"
                >×</button>
              )}
            </div>
          ))}

          {/* Add new label inline */}
          <div className="flex items-center gap-1 ml-1">
            <input
              type="text"
              value={newInput}
              onChange={e => setNewInput(e.target.value)}
              onKeyDown={e => e.key === 'Enter' && addLabel()}
              placeholder="Nhãn mới..."
              disabled={isRecording}
              className="bg-[#111827] border border-gray-700 rounded-md px-3 py-2 text-sm text-white placeholder-gray-600 w-32 focus:outline-none focus:border-blue-500 disabled:opacity-50 transition-colors"
            />
            <button
              onClick={addLabel}
              disabled={isRecording || !newInput.trim()}
              className="bg-gray-700 hover:bg-gray-600 border border-gray-600 text-white w-9 h-9 rounded-md text-lg font-medium disabled:opacity-50 flex items-center justify-center transition-colors"
            >+</button>
          </div>
        </div>
      </div>

      {/* Controls */}
      <div className="flex items-center gap-3">
        <button
          onClick={handleStart}
          disabled={isRecording || !selected}
          className="bg-blue-600 hover:bg-blue-500 text-white px-5 py-2.5 rounded-md font-semibold text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
        >▶ Bắt đầu</button>
        <button
          onClick={handleStop}
          disabled={!isRecording}
          className="bg-red-700 hover:bg-red-600 text-white px-5 py-2.5 rounded-md font-semibold text-sm disabled:opacity-50 disabled:cursor-not-allowed transition-opacity"
        >⏹ Dừng</button>

        {isRecording && (
          <div className="flex items-center gap-2 ml-auto text-sm font-medium">
            <span className="w-2.5 h-2.5 bg-red-500 rounded-full animate-pulse" />
            <span className="text-gray-400">Đang ghi:</span>
            <span className="text-white">{currentLabel}</span>
            {sessionName && <span className="text-gray-500">({sessionName})</span>}
          </div>
        )}
      </div>
    </div>
  );
}
