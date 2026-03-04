/**
 * Opportunity Priority Matrix — 2D scatter chart for the report view.
 * X axis: Customer Priority (mean evidence signal confidence — real market data only).
 * Y axis: MathWorks Business Value (Validator confidence_score).
 * Dots are colored by opportunity confidence level and labeled with product names.
 */

import {
  ResponsiveContainer,
  ScatterChart,
  Scatter,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ReferenceLine,
  ReferenceArea,
  LabelList,
  type TooltipProps,
} from 'recharts';

export interface ChartOpportunity {
  product_name: string;
  customer_priority_score: number; // 0.0–1.0
  seller_value_score: number;      // 0.0–1.0
  estimated_value: string;
  confidence: 'low' | 'medium' | 'high';
}

interface ChartPoint {
  x: number;
  y: number;
  name: string;
  shortName: string;
  value: string;
  confidence: 'low' | 'medium' | 'high';
}

interface OpportunityMatrixProps {
  opportunities: ChartOpportunity[];
}

const CONFIDENCE_COLORS: Record<string, string> = {
  high:   '#0d9488', // teal-600
  medium: '#f59e0b', // amber-500
  low:    '#f43f5e', // rose-500
};

// ─── Custom Dot ───────────────────────────────────────────────────────────────
function CustomDot(props: {
  cx?: number;
  cy?: number;
  payload?: ChartPoint;
}) {
  const { cx = 0, cy = 0, payload } = props;
  const color = CONFIDENCE_COLORS[payload?.confidence ?? 'medium'];
  return (
    <circle
      cx={cx}
      cy={cy}
      r={10}
      fill={color}
      fillOpacity={0.85}
      stroke="white"
      strokeWidth={2}
    />
  );
}

// ─── Progress Bar helper ───────────────────────────────────────────────────────
function MiniBar({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const filled = Math.round(pct / 10);
  return (
    <span className="font-mono text-xs text-zinc-400">
      {'█'.repeat(filled)}{'░'.repeat(10 - filled)}
      {' '}
      <span className="text-zinc-600 font-semibold">{pct}%</span>
    </span>
  );
}

// ─── Custom Tooltip ────────────────────────────────────────────────────────────
// OVERLAP_THRESHOLD: ~21px on a 420px chart — catches visually stacked dots
const OVERLAP_THRESHOLD = 0.05;

function CustomTooltip({
  active,
  payload,
  allPoints = [],
}: TooltipProps<number, string> & { allPoints?: ChartPoint[] }) {
  if (!active || !payload?.length) return null;
  const hovered = payload[0].payload as ChartPoint;

  // Find all points visually overlapping with the hovered dot
  const nearby = allPoints.filter(p => {
    const dx = p.x - hovered.x;
    const dy = p.y - hovered.y;
    return Math.sqrt(dx * dx + dy * dy) < OVERLAP_THRESHOLD;
  });
  const pointsToShow = nearby.length > 0 ? nearby : [hovered];

  return (
    <div
      className="bg-white border border-zinc-200 rounded-lg shadow-lg p-3 text-xs"
      style={{ minWidth: 220 }}
    >
      {pointsToShow.map((d, i) => {
        const confColor = CONFIDENCE_COLORS[d.confidence];
        const confLabel = d.confidence.charAt(0).toUpperCase() + d.confidence.slice(1);
        return (
          <div key={i} className={i > 0 ? 'mt-3 pt-3 border-t border-zinc-100' : ''}>
            <p className="font-semibold text-zinc-900 mb-2 leading-snug">{d.name}</p>
            <div className="space-y-1.5">
              <div className="flex items-center justify-between gap-4">
                <span className="text-zinc-500 whitespace-nowrap">Customer Priority</span>
                <MiniBar value={d.x} />
              </div>
              <div className="flex items-center justify-between gap-4">
                <span className="text-zinc-500 whitespace-nowrap">MathWorks Value</span>
                <MiniBar value={d.y} />
              </div>
              {d.value && (
                <div className="flex items-center justify-between gap-4 pt-1 border-t border-zinc-100">
                  <span className="text-zinc-500">Est. Value</span>
                  <span className="font-medium text-zinc-700">{d.value}</span>
                </div>
              )}
              <div className="flex items-center justify-between gap-4">
                <span className="text-zinc-500">Confidence</span>
                <span className="flex items-center gap-1 font-medium" style={{ color: confColor }}>
                  <span
                    className="inline-block w-2 h-2 rounded-full"
                    style={{ backgroundColor: confColor }}
                  />
                  {confLabel}
                </span>
              </div>
            </div>
          </div>
        );
      })}
    </div>
  );
}

// ─── Quadrant Corner Labels via SVG ───────────────────────────────────────────
function QuadrantLabels(props: { width?: number; height?: number; margin?: { top: number; right: number; bottom: number; left: number } }) {
  const { width = 0, height = 0, margin = { top: 30, right: 60, bottom: 40, left: 40 } } = props;
  const innerW = width - margin.left - margin.right;
  const innerH = height - margin.top - margin.bottom;

  if (innerW <= 0 || innerH <= 0) return null;

  // x=0.5 midpoint in pixel space, y-axis is inverted in SVG
  const midX = margin.left + innerW / 2;
  const midY = margin.top + innerH / 2;

  const pad = 8;

  return (
    <g style={{ pointerEvents: 'none' }}>
      {/* Top-right: Focus First ★ — teal, bold */}
      <text
        x={margin.left + innerW - pad}
        y={margin.top + pad + 11}
        textAnchor="end"
        fill="#0d9488"
        fontSize={11}
        fontWeight={600}
      >
        Focus First ★
      </text>
      {/* Top-left: Build Awareness */}
      <text
        x={margin.left + pad}
        y={margin.top + pad + 11}
        textAnchor="start"
        fill="#a1a1aa"
        fontSize={11}
      >
        Build Awareness
      </text>
      {/* Bot-right: Latent Need */}
      <text
        x={margin.left + innerW - pad}
        y={margin.top + innerH - pad}
        textAnchor="end"
        fill="#d97706"
        fontSize={11}
      >
        Latent Need
      </text>
      {/* Bot-left: Monitor */}
      <text
        x={margin.left + pad}
        y={margin.top + innerH - pad}
        textAnchor="start"
        fill="#a1a1aa"
        fontSize={11}
      >
        Monitor
      </text>
      {/* Mid-line labels */}
      <text
        x={midX}
        y={margin.top - 8}
        textAnchor="middle"
        fill="#71717a"
        fontSize={9}
      />
      {/* invisible anchor so recharts doesn't complain about unused */}
      <rect x={midX} y={midY} width={0} height={0} />
    </g>
  );
}

// ─── Main Component ────────────────────────────────────────────────────────────
export function OpportunityMatrix({ opportunities }: OpportunityMatrixProps) {
  const points: ChartPoint[] = opportunities.map(o => ({
    x: o.customer_priority_score,
    y: o.seller_value_score,
    name: o.product_name,
    shortName: o.product_name.length > 18 ? o.product_name.slice(0, 16) + '…' : o.product_name,
    value: o.estimated_value,
    confidence: o.confidence,
  }));

  const axisTicks = [0, 0.25, 0.5, 0.75, 1.0];
  const margin = { top: 30, right: 60, bottom: 50, left: 50 };

  return (
    <div className="bg-white rounded-xl border border-zinc-200 shadow-sm p-6 mb-8">
      {/* Header */}
      <div className="mb-3">
        <h2 className="text-lg font-semibold text-zinc-900">Opportunity Priority Matrix</h2>
        <p className="text-sm text-zinc-500 mt-0.5">
          Plotted by customer market signals (X) vs. MathWorks confidence score (Y)
        </p>
      </div>

      {/* Legend */}
      <div className="flex items-center gap-6 mb-4 text-xs text-zinc-500">
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full inline-block" style={{ backgroundColor: CONFIDENCE_COLORS.high }} />
          High confidence
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full inline-block" style={{ backgroundColor: CONFIDENCE_COLORS.medium }} />
          Medium
        </span>
        <span className="flex items-center gap-1.5">
          <span className="w-3 h-3 rounded-full inline-block" style={{ backgroundColor: CONFIDENCE_COLORS.low }} />
          Low
        </span>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={420}>
        <ScatterChart margin={margin}>
          {/* Quadrant background fills */}
          <ReferenceArea x1={0}   x2={0.5} y1={0.5} y2={1}   fill="#eff6ff" fillOpacity={0.6} />
          <ReferenceArea x1={0.5} x2={1}   y1={0.5} y2={1}   fill="#f0fdf4" fillOpacity={0.6} />
          <ReferenceArea x1={0}   x2={0.5} y1={0}   y2={0.5} fill="#fafafa" fillOpacity={0.6} />
          <ReferenceArea x1={0.5} x2={1}   y1={0}   y2={0.5} fill="#fffbeb" fillOpacity={0.6} />

          <CartesianGrid strokeDasharray="3 3" stroke="#e4e4e7" />

          <XAxis
            type="number"
            dataKey="x"
            domain={[0, 1]}
            ticks={axisTicks}
            tickFormatter={v => `${Math.round(v * 100)}%`}
            tick={{ fontSize: 11, fill: '#71717a' }}
            label={{
              value: 'Customer Priority →',
              position: 'insideBottom',
              offset: -10,
              fontSize: 12,
              fill: '#52525b',
            }}
          />

          <YAxis
            type="number"
            dataKey="y"
            domain={[0, 1]}
            ticks={axisTicks}
            tickFormatter={v => `${Math.round(v * 100)}%`}
            tick={{ fontSize: 11, fill: '#71717a' }}
            label={{
              value: 'MathWorks Business Value →',
              angle: -90,
              position: 'insideLeft',
              offset: 10,
              fontSize: 12,
              fill: '#52525b',
            }}
          />

          {/* Dashed mid-lines */}
          <ReferenceLine x={0.5} stroke="#a1a1aa" strokeDasharray="4 4" strokeWidth={1.5} />
          <ReferenceLine y={0.5} stroke="#a1a1aa" strokeDasharray="4 4" strokeWidth={1.5} />

          <Tooltip content={(props) => <CustomTooltip {...props} allPoints={points} />} />

          {/* Quadrant corner labels rendered as custom SVG overlay */}
          {/* @ts-expect-error recharts Customized accepts any component */}
          <QuadrantLabels />

          <Scatter
            data={points}
            shape={<CustomDot />}
          >
            <LabelList
              dataKey="shortName"
              position="top"
              offset={14}
              style={{ fontSize: 11, fill: '#52525b' }}
            />
          </Scatter>
        </ScatterChart>
      </ResponsiveContainer>
    </div>
  );
}
