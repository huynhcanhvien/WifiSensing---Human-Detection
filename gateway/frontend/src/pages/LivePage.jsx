import { StatusCards }   from '../components/StatusCards';
import { CSIHeatmap }    from '../components/CSIHeatmap';
import { LiveCharts }    from '../components/LiveCharts';
import { LabelingPanel } from '../components/LabelingPanel';

const MAC0 = 'd4e9f4a36480';
const MAC1 = 'd4e9f4a43c18';

export function LivePage({ wsStatus, aiResult, currentLabel, csiBuffersRef, labels, setLabels, send }) {
  return (
    <div>
      <StatusCards wsStatus={wsStatus} aiResult={aiResult} />

      <div className="grid grid-cols-2 gap-4 mb-4">
        <CSIHeatmap mac={MAC0} label="Heatmap CSI — MAC 0" badge={MAC0} csiBuffersRef={csiBuffersRef} />
        <CSIHeatmap mac={MAC1} label="Heatmap CSI — MAC 1" badge={MAC1} csiBuffersRef={csiBuffersRef} />
      </div>

      <LiveCharts csiBuffersRef={csiBuffersRef} />

      <LabelingPanel
        labels={labels}
        setLabels={setLabels}
        currentLabel={currentLabel}
        send={send}
      />
    </div>
  );
}
