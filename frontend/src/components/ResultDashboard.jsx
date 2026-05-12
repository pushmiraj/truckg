import { useState } from 'react'
import {
  MapPin, ArrowRight, TrendingUp, ShieldCheck, Brain, BarChart3,
  Clock, Truck, Package, ChevronDown, ChevronUp, AlertTriangle,
  CheckCircle, XCircle, Award,
} from 'lucide-react'

// ---------------------------------------------------------------------------
// Utilities
// ---------------------------------------------------------------------------

const inr = (n) =>
  new Intl.NumberFormat('en-IN', { style: 'currency', currency: 'INR', maximumFractionDigits: 0 }).format(n)

const pct = (n, digits = 1) => `${(n * 100).toFixed(digits)}%`

function confLabel(c) {
  if (c >= 0.75) return { text: 'HIGH', cls: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30' }
  if (c >= 0.45) return { text: 'MEDIUM', cls: 'bg-amber-500/15 text-amber-400 border-amber-500/30' }
  return               { text: 'LOW',    cls: 'bg-red-500/15 text-red-400 border-red-500/30' }
}

function riskLabel(p) {
  if (p <= 0.20) return { text: 'LOW RISK',  cls: 'bg-emerald-500/15 text-emerald-400 border-emerald-500/30', Icon: CheckCircle }
  if (p <= 0.45) return { text: 'MODERATE',  cls: 'bg-amber-500/15 text-amber-400 border-amber-500/30',  Icon: AlertTriangle }
  return               { text: 'HIGH RISK', cls: 'bg-red-500/15 text-red-400 border-red-500/30',       Icon: XCircle }
}

function pReturnLabel(p) {
  if (p >= 0.60) return { text: 'HIGH', cls: 'text-emerald-400' }
  if (p >= 0.35) return { text: 'MED',  cls: 'text-amber-400'   }
  return               { text: 'LOW',  cls: 'text-red-400'      }
}

// ---------------------------------------------------------------------------
// Price layer step card
// ---------------------------------------------------------------------------

const LAYER_META = [
  { key: 'operational',   label: 'Layer 1',   title: 'Operational Base',   icon: Truck,      color: 'from-slate-700 to-slate-800',       accent: 'text-slate-300', desc: 'Distance × Weight × Vehicle Rate' },
  { key: 'risk_adjusted', label: 'Layer 2',   title: 'Risk Adjusted',      icon: ShieldCheck, color: 'from-orange-900/50 to-slate-800',  accent: 'text-orange-300', desc: 'Base × (1 + empty-return risk)' },
  { key: 'ml_optimized',  label: 'Layer 3',   title: 'ML Optimised',       icon: Brain,       color: 'from-indigo-900/50 to-slate-800',  accent: 'text-indigo-300', desc: 'Risk × AI efficiency factor' },
  { key: 'market_price',  label: 'Layer 4',   title: 'Market Snapped',     icon: BarChart3,   color: 'from-blue-900/50 to-slate-800',    accent: 'text-blue-300',   desc: 'Clamped ± 15% corridor average' },
  { key: 'final_price',   label: 'Layer 5 ★', title: 'Final Quoted Price', icon: Award,       color: 'from-emerald-900/50 to-slate-800', accent: 'text-emerald-300', desc: 'Max(Market, Base + 10% floor)' },
]

function LayerCard({ meta, value, prev, isLast }) {
  const Icon = meta.icon
  const delta = prev != null ? value - prev : 0
  const sign  = delta >= 0 ? '+' : ''

  return (
    <div className={`relative flex-1 min-w-0 rounded-xl border border-slate-700/70
                     bg-gradient-to-b ${meta.color} p-4
                     ${isLast ? 'ring-2 ring-emerald-500/40 shadow-lg' : ''}`}>
      <div className="flex items-center gap-2 mb-3">
        <div className={`p-1.5 rounded-lg bg-slate-700/60`}>
          <Icon size={13} className={meta.accent} />
        </div>
        <div>
          <div className={`text-[10px] font-semibold tracking-widest uppercase ${meta.accent} opacity-70`}>
            {meta.label}
          </div>
          <div className="text-xs font-semibold text-white leading-tight">{meta.title}</div>
        </div>
      </div>

      <div className={`text-lg font-bold font-mono ${isLast ? 'text-emerald-300' : 'text-white'}`}>
        {inr(value)}
      </div>

      {prev != null && (
        <div className={`text-[11px] mt-0.5 font-mono
          ${delta > 0 ? 'text-orange-400/80' : delta < 0 ? 'text-emerald-400/80' : 'text-slate-500'}`}>
          {sign}{inr(Math.abs(delta))}
        </div>
      )}

      <p className="text-[10px] text-slate-500 mt-2 leading-snug">{meta.desc}</p>
    </div>
  )
}

// ---------------------------------------------------------------------------
// Metric card
// ---------------------------------------------------------------------------

function MetricCard({ icon: Icon, label, value, sub, badge, accentClass = 'text-indigo-400' }) {
  return (
    <div className="card p-4">
      <div className="flex items-start justify-between mb-3">
        <div className="p-2 bg-slate-700/60 rounded-lg">
          <Icon size={15} className={accentClass} />
        </div>
        {badge && (
          <span className={`badge border text-[10px] ${badge.cls}`}>
            {badge.Icon && <badge.Icon size={10} />}
            {badge.text}
          </span>
        )}
      </div>
      <div className={`text-2xl font-bold font-mono ${accentClass} mb-0.5`}>{value}</div>
      <div className="text-xs text-slate-400">{label}</div>
      {sub && <div className="text-[11px] text-slate-500 mt-1">{sub}</div>}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Main dashboard
// ---------------------------------------------------------------------------

export default function ResultDashboard({ result }) {
  const [expanded, setExpanded] = useState(false)

  const { route, ml, pricing, trip } = result
  const risk   = riskLabel(ml.delay_probability)
  const conf   = confLabel(ml.confidence)
  const pret   = pReturnLabel(ml.p_return)

  const prices = LAYER_META.map(m => pricing[m.key])
  const savings = pricing.ml_optimized < pricing.risk_adjusted
    ? pricing.risk_adjusted - pricing.ml_optimized
    : 0

  return (
    <div className="space-y-6 animate-in fade-in duration-300" style={{ '--tw-enter-duration': '300ms' }}>

      {/* Route header */}
      <div className="card p-5">
        <div className="flex flex-wrap items-center gap-x-4 gap-y-2 justify-between">
          <div className="flex items-center gap-3 flex-wrap">
            <div className="flex items-center gap-2">
              <MapPin size={14} className="text-indigo-400" />
              <span className="font-semibold text-white">{route.origin}</span>
            </div>
            <ArrowRight size={14} className="text-slate-500" />
            <div className="flex items-center gap-2">
              <MapPin size={14} className="text-emerald-400" />
              <span className="font-semibold text-white">{route.destination}</span>
            </div>
          </div>
          <div className="flex items-center gap-3 text-xs text-slate-400">
            <span className="flex items-center gap-1.5">
              <TrendingUp size={12} className="text-indigo-400" />
              {route.distance_km.toLocaleString('en-IN')} km
            </span>
            <span className="text-slate-600">·</span>
            <span className="flex items-center gap-1.5">
              <Truck size={12} className="text-indigo-400" />
              {trip.weight_tons}t · {trip.vehicle_class}
            </span>
            <span className="text-slate-600">·</span>
            <span className="capitalize text-slate-400">{route.region}</span>
          </div>
        </div>
      </div>

      {/* Final price + confidence hero */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <div className="sm:col-span-2 card p-6 bg-gradient-to-br from-emerald-900/20 to-slate-800
                        border-emerald-700/30 glow-indigo">
          <div className="text-xs font-semibold tracking-widest uppercase text-emerald-400/70 mb-2">
            Final Quoted Price
          </div>
          <div className="text-4xl font-bold font-mono text-emerald-300 mb-3">
            {inr(pricing.final_price)}
          </div>
          <div className="flex items-center gap-3 flex-wrap">
            <span className={`badge border text-xs ${conf.cls}`}>
              {conf.text} CONFIDENCE
            </span>
            <span className={`badge border text-xs ${risk.cls}`}>
              <risk.Icon size={10} />
              {risk.text}
            </span>
            {pricing.capped && (
              <span className="badge bg-blue-500/15 text-blue-400 border border-blue-500/30 text-xs">
                MARKET CAPPED
              </span>
            )}
          </div>
          {savings > 0 && (
            <p className="text-xs text-emerald-400/70 mt-3">
              AI optimisation saved {inr(savings)} vs raw risk-adjusted price
            </p>
          )}
        </div>

        <div className="card p-5 flex flex-col justify-between">
          <div>
            <div className="text-[10px] font-semibold tracking-widest uppercase text-slate-400 mb-1">
              AI Risk Score
            </div>
            <div className="text-3xl font-bold font-mono text-white">
              {pct(ml.delay_probability)}
            </div>
            <div className="text-xs text-slate-400 mt-0.5">delay probability</div>
          </div>
          <div className="mt-4">
            <div className="flex justify-between text-[10px] text-slate-500 mb-1">
              <span>Low risk</span><span>High risk</span>
            </div>
            <div className="w-full bg-slate-700 rounded-full h-2">
              <div
                className={`h-2 rounded-full transition-all duration-500
                  ${ml.delay_probability <= 0.2 ? 'bg-emerald-500'
                    : ml.delay_probability <= 0.45 ? 'bg-amber-500'
                    : 'bg-red-500'}`}
                style={{ width: pct(ml.delay_probability) }}
              />
            </div>
          </div>
        </div>
      </div>

      {/* 5-layer breakdown */}
      <div className="card p-5">
        <div className="flex items-center gap-2 mb-5">
          <BarChart3 size={15} className="text-indigo-400" />
          <h3 className="font-semibold text-white text-sm">5-Layer Pricing Breakdown</h3>
          <span className="text-xs text-slate-500 ml-auto">Price evolution through each layer</span>
        </div>

        {/* Desktop: horizontal flow */}
        <div className="hidden sm:flex items-stretch gap-2">
          {LAYER_META.map((meta, i) => (
            <div key={meta.key} className="flex items-center gap-2 flex-1 min-w-0">
              <LayerCard
                meta={meta}
                value={prices[i]}
                prev={i > 0 ? prices[i - 1] : null}
                isLast={i === LAYER_META.length - 1}
              />
              {i < LAYER_META.length - 1 && (
                <ArrowRight size={14} className="text-slate-600 shrink-0" />
              )}
            </div>
          ))}
        </div>

        {/* Mobile: vertical stack */}
        <div className="sm:hidden space-y-2">
          {LAYER_META.map((meta, i) => (
            <div key={meta.key} className="flex items-center gap-3">
              <div className="flex-1">
                <LayerCard
                  meta={meta}
                  value={prices[i]}
                  prev={i > 0 ? prices[i - 1] : null}
                  isLast={i === LAYER_META.length - 1}
                />
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* ML intelligence metrics */}
      <div>
        <div className="flex items-center gap-2 mb-3">
          <Brain size={15} className="text-indigo-400" />
          <h3 className="font-semibold text-white text-sm">ML Intelligence Panel</h3>
        </div>
        <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
          <MetricCard
            icon={AlertTriangle}
            label="Delay Probability"
            value={pct(ml.delay_probability)}
            sub={`${pct(ml.on_time_probability)} on-time`}
            badge={risk}
            accentClass={ml.delay_probability <= 0.2 ? 'text-emerald-400'
              : ml.delay_probability <= 0.45 ? 'text-amber-400' : 'text-red-400'}
          />
          <MetricCard
            icon={Package}
            label="Return Load Prob."
            value={pct(ml.p_return)}
            sub={`${pret.text} backhaul availability`}
            accentClass={pret.cls}
          />
          <MetricCard
            icon={Clock}
            label="Est. Delivery Time"
            value={`${ml.delivery_hours.toFixed(1)}h`}
            sub={`${(ml.delivery_hours / 24).toFixed(1)} days`}
            accentClass="text-blue-400"
          />
          <MetricCard
            icon={Award}
            label="Model Confidence"
            value={pct(ml.confidence)}
            sub="certainty of prediction"
            badge={conf}
            accentClass={ml.confidence >= 0.75 ? 'text-emerald-400'
              : ml.confidence >= 0.45 ? 'text-amber-400' : 'text-red-400'}
          />
        </div>
      </div>

      {/* Expandable technical details */}
      <div className="card overflow-hidden">
        <button
          onClick={() => setExpanded(e => !e)}
          className="w-full flex items-center justify-between px-5 py-4
                     text-sm text-slate-400 hover:text-white transition-colors">
          <span className="flex items-center gap-2">
            <BarChart3 size={14} />
            Technical Details
          </span>
          {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
        </button>

        {expanded && (
          <div className="border-t border-slate-700 px-5 py-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-6 text-xs">
              {/* Pricing parameters */}
              <div>
                <div className="text-slate-400 font-semibold mb-3 uppercase tracking-wider text-[10px]">
                  Pricing Parameters
                </div>
                <table className="w-full">
                  <tbody className="space-y-1">
                    {[
                      ['Corridor Average',   inr(pricing.corridor_avg)],
                      ['Risk Factor',        `${(pricing.risk_factor * 100).toFixed(2)}%`],
                      ['ML Efficiency',      `${(pricing.efficiency * 100).toFixed(1)}%`],
                      ['Market Capped',      pricing.capped ? 'Yes' : 'No'],
                      ['Margin Floor',       inr(pricing.operational * 1.10)],
                    ].map(([k, v]) => (
                      <tr key={k} className="border-b border-slate-700/40">
                        <td className="py-1.5 text-slate-500">{k}</td>
                        <td className="py-1.5 text-right font-mono text-white">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>

              {/* ML model outputs */}
              <div>
                <div className="text-slate-400 font-semibold mb-3 uppercase tracking-wider text-[10px]">
                  Raw ML Outputs
                </div>
                <table className="w-full">
                  <tbody>
                    {[
                      ['Delay Probability',   ml.delay_probability.toFixed(4)],
                      ['On-Time Probability', ml.on_time_probability.toFixed(4)],
                      ['Return Load (p_return)', ml.p_return.toFixed(4)],
                      ['Delivery Hours',      ml.delivery_hours.toFixed(2) + ' h'],
                      ['Confidence',          ml.confidence.toFixed(4)],
                    ].map(([k, v]) => (
                      <tr key={k} className="border-b border-slate-700/40">
                        <td className="py-1.5 text-slate-500">{k}</td>
                        <td className="py-1.5 text-right font-mono text-indigo-300">{v}</td>
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* Route info */}
            <div className="mt-4 pt-4 border-t border-slate-700/40 flex flex-wrap gap-x-6 gap-y-1 text-[11px] text-slate-500">
              <span>Distance: <span className="text-slate-300 font-mono">{route.distance_km} km</span></span>
              <span>Region: <span className="text-slate-300">{route.region}</span></span>
              <span>Weight: <span className="text-slate-300 font-mono">{trip.weight_tons} t</span></span>
              <span>Class: <span className="text-slate-300">{trip.vehicle_class}</span></span>
              <span>Date: <span className="text-slate-300">{trip.date?.split('T')[0] || '—'}</span></span>
            </div>
          </div>
        )}
      </div>

    </div>
  )
}
