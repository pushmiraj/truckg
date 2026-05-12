import { useState, useEffect } from 'react'
import { Truck, LogOut, Zap } from 'lucide-react'
import { supabase, DEV_MODE } from './supabase'
import AuthPage from './components/AuthPage'
import InputForm from './components/InputForm'
import ResultDashboard from './components/ResultDashboard'

export default function App() {
  const [session, setSession] = useState(DEV_MODE ? { user: { email: 'dev@freightiq.local' } } : null)
  const [result,  setResult]  = useState(null)
  const [loading, setLoading] = useState(!DEV_MODE)

  useEffect(() => {
    if (DEV_MODE) return
    supabase.auth.getSession().then(({ data: { session } }) => {
      setSession(session)
      setLoading(false)
    })
    const { data: { subscription } } = supabase.auth.onAuthStateChange((_event, session) => {
      setSession(session)
      setResult(null)
    })
    return () => subscription.unsubscribe()
  }, [])

  const handleLogout = async () => {
    if (supabase) await supabase.auth.signOut()
    setSession(null)
    setResult(null)
  }

  if (loading) {
    return (
      <div className="min-h-screen bg-slate-950 flex items-center justify-center">
        <div className="w-8 h-8 border-2 border-indigo-500 border-t-transparent rounded-full animate-spin" />
      </div>
    )
  }

  if (!session) return <AuthPage onSession={setSession} />

  return (
    <div className="min-h-screen bg-slate-950">
      {/* Top navigation */}
      <header className="border-b border-slate-800 bg-slate-900/80 backdrop-blur-sm sticky top-0 z-40">
        <div className="max-w-6xl mx-auto px-6 h-14 flex items-center justify-between">
          <div className="flex items-center gap-2.5">
            <div className="w-7 h-7 bg-indigo-600 rounded-lg flex items-center justify-center">
              <Truck size={15} className="text-white" />
            </div>
            <span className="font-bold text-white tracking-tight">FreightIQ</span>
            {DEV_MODE && (
              <span className="badge bg-amber-500/15 text-amber-400 border border-amber-500/30 ml-1">
                Dev Mode
              </span>
            )}
          </div>

          <div className="flex items-center gap-4">
            <div className="flex items-center gap-2">
              <div className="w-1.5 h-1.5 bg-emerald-400 rounded-full animate-pulse" />
              <span className="text-slate-400 text-xs font-mono">
                {session.user?.email || 'demo'}
              </span>
            </div>
            {!DEV_MODE && (
              <button onClick={handleLogout}
                className="text-slate-400 hover:text-white transition-colors p-1.5 rounded-lg hover:bg-slate-700">
                <LogOut size={15} />
              </button>
            )}
          </div>
        </div>
      </header>

      {/* Hero strip */}
      <div className="border-b border-slate-800/50 bg-gradient-to-r from-slate-900 to-slate-950">
        <div className="max-w-6xl mx-auto px-6 py-8">
          <div className="flex items-start gap-3">
            <div className="mt-0.5 p-2 bg-indigo-600/20 rounded-lg border border-indigo-500/30">
              <Zap size={18} className="text-indigo-400" />
            </div>
            <div>
              <h1 className="text-xl font-bold text-white">
                AI-Driven Freight Pricing
              </h1>
              <p className="text-slate-400 text-sm mt-0.5">
                Risk-quantified dynamic quotes powered by XGBoost · 3-model ML stack · 5-layer pricing engine
              </p>
            </div>
          </div>
        </div>
      </div>

      {/* Main content */}
      <main className="max-w-6xl mx-auto px-6 py-8 space-y-8">
        <InputForm session={session} onResult={(r) => { setResult(r); window.scrollTo({ top: 400, behavior: 'smooth' }) }} />
        {result && <ResultDashboard result={result} />}
      </main>

      <footer className="border-t border-slate-800 mt-16 py-6 text-center">
        <p className="text-slate-500 text-xs">
          FreightIQ · XGBoost · 6,880 GPS pings · Tamil Nadu · Karnataka · Pondicherry · Maharashtra
        </p>
      </footer>
    </div>
  )
}
