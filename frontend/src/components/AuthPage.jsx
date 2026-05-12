import { useState } from 'react'
import { Truck, Mail, Lock, UserPlus, LogIn, AlertCircle } from 'lucide-react'
import { supabase } from '../supabase'

export default function AuthPage({ onSession }) {
  const [mode,     setMode]     = useState('login')     // 'login' | 'signup'
  const [email,    setEmail]    = useState('')
  const [password, setPassword] = useState('')
  const [loading,  setLoading]  = useState(false)
  const [error,    setError]    = useState('')
  const [success,  setSuccess]  = useState('')

  const handleSubmit = async (e) => {
    e.preventDefault()
    setError('')
    setSuccess('')
    setLoading(true)

    try {
      if (mode === 'login') {
        const { data, error } = await supabase.auth.signInWithPassword({ email, password })
        if (error) throw error
        onSession(data.session)
      } else {
        const { error } = await supabase.auth.signUp({ email, password })
        if (error) throw error
        setSuccess('Account created! Check your email to confirm, then sign in.')
        setMode('login')
      }
    } catch (err) {
      setError(err.message || 'Something went wrong')
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-slate-950 flex flex-col items-center justify-center px-4">
      {/* Background glow */}
      <div className="absolute inset-0 overflow-hidden pointer-events-none">
        <div className="absolute -top-40 left-1/2 -translate-x-1/2 w-[600px] h-[600px]
                        bg-indigo-600/10 rounded-full blur-3xl" />
      </div>

      <div className="relative w-full max-w-md">
        {/* Logo */}
        <div className="text-center mb-8">
          <div className="inline-flex items-center justify-center w-14 h-14
                          bg-indigo-600 rounded-2xl mb-4 shadow-lg glow-indigo">
            <Truck size={28} className="text-white" />
          </div>
          <h1 className="text-2xl font-bold text-white">FreightIQ</h1>
          <p className="text-slate-400 text-sm mt-1">AI-Driven Logistics Pricing Engine</p>
        </div>

        {/* Card */}
        <div className="card p-8 shadow-2xl">
          {/* Tab toggle */}
          <div className="flex mb-6 bg-slate-700/50 rounded-lg p-1">
            {(['login', 'signup']).map((m) => (
              <button key={m}
                onClick={() => { setMode(m); setError(''); setSuccess('') }}
                className={`flex-1 py-2 text-sm font-medium rounded-md transition-all duration-150
                  ${mode === m
                    ? 'bg-slate-600 text-white shadow-sm'
                    : 'text-slate-400 hover:text-white'
                  }`}>
                {m === 'login' ? 'Sign In' : 'Sign Up'}
              </button>
            ))}
          </div>

          {/* Alerts */}
          {error && (
            <div className="mb-4 flex items-start gap-2 bg-red-500/10 border border-red-500/30
                            text-red-400 text-sm px-3.5 py-3 rounded-lg">
              <AlertCircle size={15} className="shrink-0 mt-0.5" />
              <span>{error}</span>
            </div>
          )}
          {success && (
            <div className="mb-4 bg-emerald-500/10 border border-emerald-500/30
                            text-emerald-400 text-sm px-3.5 py-3 rounded-lg">
              {success}
            </div>
          )}

          <form onSubmit={handleSubmit} className="space-y-4">
            {/* Email */}
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">
                Email address
              </label>
              <div className="relative">
                <Mail size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
                <input type="email" required autoComplete="email"
                  value={email} onChange={e => setEmail(e.target.value)}
                  placeholder="you@company.com"
                  className="input-field pl-10" />
              </div>
            </div>

            {/* Password */}
            <div>
              <label className="block text-xs font-medium text-slate-300 mb-1.5">
                Password
              </label>
              <div className="relative">
                <Lock size={15} className="absolute left-3.5 top-1/2 -translate-y-1/2 text-slate-400" />
                <input type="password" required autoComplete="current-password"
                  value={password} onChange={e => setPassword(e.target.value)}
                  placeholder="••••••••"
                  className="input-field pl-10" />
              </div>
            </div>

            <button type="submit" disabled={loading} className="btn-primary w-full mt-2 flex items-center justify-center gap-2">
              {loading
                ? <div className="w-4 h-4 border-2 border-white/30 border-t-white rounded-full animate-spin" />
                : mode === 'login'
                  ? <><LogIn size={15} /> Sign In</>
                  : <><UserPlus size={15} /> Create Account</>
              }
            </button>
          </form>
        </div>

        <p className="text-center text-slate-500 text-xs mt-6">
          Indian freight corridors · Tamil Nadu · Karnataka · Maharashtra
        </p>
      </div>
    </div>
  )
}
