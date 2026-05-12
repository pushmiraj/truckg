import { useState, useRef, useEffect, useCallback } from 'react'
import { MapPin, Weight, Truck, Calendar, ArrowRight, Loader2, AlertCircle, X } from 'lucide-react'
import { supabase, DEV_MODE } from '../supabase'

// ---------------------------------------------------------------------------
// City autocomplete input
// ---------------------------------------------------------------------------

function CityInput({ label, placeholder, value, onChange }) {
  const [query,       setQuery]       = useState(value?.name || '')
  const [suggestions, setSuggestions] = useState([])
  const [open,        setOpen]        = useState(false)
  const [fetching,    setFetching]    = useState(false)
  const timer = useRef(null)
  const containerRef = useRef(null)

  // Close on outside click
  useEffect(() => {
    const handler = (e) => {
      if (containerRef.current && !containerRef.current.contains(e.target)) setOpen(false)
    }
    document.addEventListener('mousedown', handler)
    return () => document.removeEventListener('mousedown', handler)
  }, [])

  const handleChange = (e) => {
    const q = e.target.value
    setQuery(q)
    onChange({ name: q, coords: null, state: null })   // clear selection while typing

    clearTimeout(timer.current)
    if (q.length < 2) { setSuggestions([]); setOpen(false); return }

    timer.current = setTimeout(async () => {
      setFetching(true)
      try {
        const res  = await fetch(`/api/autocomplete?q=${encodeURIComponent(q)}`)
        const data = await res.json()
        const feats = (data.features || []).filter(f =>
          f.properties?.country?.toLowerCase().includes('india') ||
          f.properties?.country?.toLowerCase().includes('ind')
        )
        setSuggestions(feats.slice(0, 5))
        setOpen(feats.length > 0)
      } catch {
        setSuggestions([])
      }
      setFetching(false)
    }, 300)
  }

  const handleSelect = (feat) => {
    const { name, state, city, county } = feat.properties
    const [lon, lat] = feat.geometry.coordinates
    const display = [city || name, state].filter(Boolean).join(', ')
    setQuery(display)
    setSuggestions([])
    setOpen(false)
    onChange({ name: display, coords: { lat, lon }, state: state || '' })
  }

  const handleClear = () => {
    setQuery('')
    setSuggestions([])
    setOpen(false)
    onChange({ name: '', coords: null, state: null })
  }

  return (
    <div className="relative" ref={containerRef}>
      <label className="block text-xs font-medium text-slate-300 mb-1.5">
        <span className="flex items-center gap-1.5">
          <MapPin size={12} className="text-indigo-400" />
          {label}
        </span>
      </label>
      <div className="relative">
        <input
          type="text"
          value={query}
          onChange={handleChange}
          onFocus={() => suggestions.length > 0 && setOpen(true)}
          placeholder={placeholder}
          className="input-field pr-9"
          autoComplete="off"
        />
        <div className="absolute right-3 top-1/2 -translate-y-1/2">
          {fetching
            ? <Loader2 size={14} className="text-slate-400 animate-spin" />
            : query
              ? <button type="button" onClick={handleClear} className="text-slate-500 hover:text-slate-300">
                  <X size={14} />
                </button>
              : null
          }
        </div>
        {/* Selection indicator */}
        {value?.coords && (
          <div className="absolute -right-0.5 -top-0.5 w-2 h-2 bg-emerald-400 rounded-full" />
        )}
      </div>

      {/* Dropdown */}
      {open && suggestions.length > 0 && (
        <ul className="absolute z-50 w-full mt-1 bg-slate-700 border border-slate-600
                       rounded-lg shadow-xl overflow-hidden">
          {suggestions.map((feat, i) => {
            const { name, state, city, country } = feat.properties
            const primary   = city || name || 'Unknown'
            const secondary = [state, country].filter(Boolean).join(', ')
            return (
              <li key={i}
                onMouseDown={() => handleSelect(feat)}
                className="flex items-center gap-3 px-4 py-3 hover:bg-slate-600
                           cursor-pointer transition-colors border-b border-slate-600/50 last:border-0">
                <MapPin size={13} className="text-indigo-400 shrink-0" />
                <div>
                  <div className="text-sm text-white font-medium">{primary}</div>
                  <div className="text-xs text-slate-400">{secondary}</div>
                </div>
              </li>
            )
          })}
        </ul>
      )}
    </div>
  )
}

// ---------------------------------------------------------------------------
// Vehicle class options — mapped to model training categories
// ---------------------------------------------------------------------------

const VEHICLE_OPTIONS = [
  { value: 'mini',    label: 'Mini Truck',   sub: 'Up to 1 ton'   },
  { value: 'medium',  label: 'SXL / LCV',    sub: '1 – 7 tons'   },
  { value: 'heavy',   label: 'HCV Multi-Axle', sub: '7 – 20 tons' },
  { value: 'trailer', label: '24 FT Container', sub: '20+ tons'   },
]

// ---------------------------------------------------------------------------
// Main form
// ---------------------------------------------------------------------------

export default function InputForm({ session, onResult }) {
  const today = new Date().toISOString().split('T')[0]

  const [origin,  setOrigin]  = useState({ name: '', coords: null, state: null })
  const [dest,    setDest]    = useState({ name: '', coords: null, state: null })
  const [weight,  setWeight]  = useState(10)
  const [vclass,  setVclass]  = useState('medium')
  const [date,    setDate]    = useState(today)
  const [loading, setLoading] = useState(false)
  const [error,   setError]   = useState('')

  const canSubmit = origin.coords && dest.coords && weight > 0 && !loading

  const handleSubmit = async (e) => {
    e.preventDefault()
    if (!canSubmit) return
    setError('')
    setLoading(true)

    // Build auth header
    const headers = { 'Content-Type': 'application/json' }
    if (!DEV_MODE && session) {
      const { data: { session: s } } = await supabase.auth.getSession()
      if (s?.access_token) headers['Authorization'] = `Bearer ${s.access_token}`
    }

    try {
      const res = await fetch('/api/predict', {
        method:  'POST',
        headers,
        body: JSON.stringify({
          origin:             origin.name,
          destination:        dest.name,
          origin_coords:      origin.coords,
          destination_coords: dest.coords,
          origin_state:       origin.state,
          destination_state:  dest.state,
          weight_tons:        parseFloat(weight),
          vehicle_class:      vclass,
          date:               date ? new Date(date).toISOString() : '',
        }),
      })

      if (!res.ok) {
        const err = await res.json()
        throw new Error(err.detail || `Error ${res.status}`)
      }

      const data = await res.json()
      onResult(data)
    } catch (err) {
      setError(err.message || 'Network error — is the backend running on port 5000?')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="card p-6 shadow-xl">
      <div className="flex items-center gap-2 mb-6">
        <Truck size={18} className="text-indigo-400" />
        <h2 className="font-semibold text-white text-base">New Freight Quote</h2>
        <span className="badge bg-indigo-500/15 text-indigo-400 border border-indigo-500/30 ml-auto">
          AI Powered
        </span>
      </div>

      {error && (
        <div className="mb-5 flex items-start gap-2 bg-red-500/10 border border-red-500/30
                        text-red-400 text-sm px-4 py-3 rounded-lg">
          <AlertCircle size={14} className="shrink-0 mt-0.5" />
          <span>{error}</span>
        </div>
      )}

      <form onSubmit={handleSubmit} className="space-y-5">
        {/* Route row */}
        <div className="grid grid-cols-1 md:grid-cols-[1fr_auto_1fr] gap-4 items-end">
          <CityInput
            label="Origin City"
            placeholder="e.g. Chennai, Tamil Nadu"
            value={origin}
            onChange={setOrigin}
          />
          <div className="hidden md:flex items-center justify-center pb-2.5">
            <ArrowRight size={18} className="text-slate-500" />
          </div>
          <CityInput
            label="Destination City"
            placeholder="e.g. Mumbai, Maharashtra"
            value={dest}
            onChange={setDest}
          />
        </div>

        {/* Trip details row */}
        <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
          {/* Weight */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <Weight size={12} className="text-indigo-400" />
                Cargo Weight (tons)
              </span>
            </label>
            <input type="number" min="0.5" max="60" step="0.5"
              value={weight} onChange={e => setWeight(e.target.value)}
              className="input-field" />
          </div>

          {/* Vehicle class */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <Truck size={12} className="text-indigo-400" />
                Vehicle Class
              </span>
            </label>
            <select value={vclass} onChange={e => setVclass(e.target.value)}
              className="input-field">
              {VEHICLE_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>
                  {o.label} — {o.sub}
                </option>
              ))}
            </select>
          </div>

          {/* Date */}
          <div>
            <label className="block text-xs font-medium text-slate-300 mb-1.5">
              <span className="flex items-center gap-1.5">
                <Calendar size={12} className="text-indigo-400" />
                Dispatch Date
              </span>
            </label>
            <input type="date" value={date} onChange={e => setDate(e.target.value)}
              min={today}
              className="input-field" />
          </div>
        </div>

        {/* Submit */}
        <div className="flex items-center justify-between pt-1">
          <p className="text-xs text-slate-500">
            {origin.coords && dest.coords
              ? <span className="text-emerald-400">Route selected — ready to quote</span>
              : 'Select origin and destination cities using autocomplete'
            }
          </p>
          <button type="submit" disabled={!canSubmit} className="btn-primary flex items-center gap-2">
            {loading
              ? <><Loader2 size={14} className="animate-spin" /> Calculating…</>
              : <><Zap size={14} /> Get AI Quote</>
            }
          </button>
        </div>
      </form>
    </div>
  )
}

// Inline Zap import fix
function Zap({ size, className }) {
  return (
    <svg width={size} height={size} viewBox="0 0 24 24" fill="none"
      stroke="currentColor" strokeWidth={2} strokeLinecap="round" strokeLinejoin="round"
      className={className}>
      <polygon points="13 2 3 14 12 14 11 22 21 10 12 10 13 2" />
    </svg>
  )
}
