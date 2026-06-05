const STYLES = {
  info:    'bg-blue-950 border-blue-700 text-blue-200',
  success: 'bg-emerald-950 border-emerald-700 text-emerald-200',
  warning: 'bg-amber-950 border-amber-700 text-amber-200',
  error:   'bg-red-950 border-red-700 text-red-200',
};

const ICONS = { info: 'ℹ', success: '✓', warning: '⚠', error: '✕' };

export function Notification({ notifications }) {
  if (!notifications.length) return null;
  return (
    <div className="fixed top-[70px] right-4 z-50 flex flex-col gap-2 max-w-xs pointer-events-none">
      {notifications.map(n => (
        <div
          key={n.id}
          className={`border rounded-lg px-4 py-2.5 text-sm font-medium shadow-xl flex items-center gap-2 ${STYLES[n.type] || STYLES.info}`}
        >
          <span className="shrink-0 text-xs">{ICONS[n.type]}</span>
          {n.message}
        </div>
      ))}
    </div>
  );
}
