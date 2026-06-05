const TABS = [
  { id: 'live', label: 'Live' },
  { id: 'history', label: 'Lịch Sử' },
  { id: 'statistics', label: 'Thống Kê' },
];

export function Navbar({ activeTab, onTabChange }) {
  return (
    <header className="border-b border-gray-700 bg-[#111827] sticky top-0 z-40">
      <div className="max-w-7xl mx-auto px-4 h-14 flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <span className="w-2 h-2 rounded-full bg-emerald-500 animate-pulse" />
          <span className="font-semibold text-white text-sm tracking-tight">CSI Dashboard</span>
          <span className="text-gray-600 text-xs ml-1">WiFi Sensing</span>
        </div>
        <nav className="flex gap-1">
          {TABS.map(tab => (
            <button
              key={tab.id}
              onClick={() => onTabChange(tab.id)}
              className={`px-4 py-2 rounded-md text-sm font-medium transition-colors ${
                activeTab === tab.id
                  ? 'bg-gray-700 text-white'
                  : 'text-gray-400 hover:text-white hover:bg-gray-800'
              }`}
            >
              {tab.label}
            </button>
          ))}
        </nav>
      </div>
    </header>
  );
}
