import { useCallback, useState } from 'react'
import './App.css'

type ClassifyResponse = {
  label: string
  spam_probability: number
  version: string
  /** Server uses this cutoff for the verdict (reduces false positives vs a raw 0.5 split). */
  spam_threshold: number
  /** Raw score from the trained classifier before promo/urgency heuristics. */
  ml_spam_probability: number
}

const apiBase = (import.meta.env.VITE_API_BASE_URL ?? 'http://127.0.0.1:8000').replace(
  /\/$/,
  '',
)

async function classifyText(text: string): Promise<ClassifyResponse> {
  const res = await fetch(`${apiBase}/classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text }),
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    const detail =
      typeof err === 'object' && err && 'detail' in err
        ? String((err as { detail: unknown }).detail)
        : res.statusText
    throw new Error(detail || `HTTP ${res.status}`)
  }
  return res.json() as Promise<ClassifyResponse>
}

/** Matches server `email_text` training format. */
function buildClassifyPayload(fromAddr: string, subject: string, body: string): string {
  const sub = subject.trim()
  const frm = fromAddr.trim()
  const txt = body.trim()
  return `Subject: ${sub}\nFrom: ${frm}\n\n${txt}`.trim()
}

async function classifyEml(file: File): Promise<ClassifyResponse> {
  const fd = new FormData()
  fd.append('file', file)
  const res = await fetch(`${apiBase}/classify/eml`, {
    method: 'POST',
    body: fd,
  })
  if (!res.ok) {
    const err = await res.json().catch(() => ({}))
    const detail =
      typeof err === 'object' && err && 'detail' in err
        ? String((err as { detail: unknown }).detail)
        : res.statusText
    throw new Error(detail || `HTTP ${res.status}`)
  }
  return res.json() as Promise<ClassifyResponse>
}

function App() {
  const [fromAddr, setFromAddr] = useState('')
  const [subject, setSubject] = useState('')
  const [body, setBody] = useState('')
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [result, setResult] = useState<ClassifyResponse | null>(null)

  const runClassify = useCallback(async () => {
    setError(null)
    setResult(null)
    if (!fromAddr.trim() && !subject.trim() && !body.trim()) {
      setError('Enter at least one of From, Subject, or message body.')
      return
    }
    const payload = buildClassifyPayload(fromAddr, subject, body)
    setLoading(true)
    try {
      setResult(await classifyText(payload))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }, [fromAddr, subject, body])

  const onFile = useCallback(async (files: FileList | null) => {
    const file = files?.[0]
    if (!file) return
    setError(null)
    setResult(null)
    setLoading(true)
    try {
      setResult(await classifyEml(file))
    } catch (e) {
      setError(e instanceof Error ? e.message : 'Request failed')
    } finally {
      setLoading(false)
    }
  }, [])

  const pct = result ? Math.round(result.spam_probability * 1000) / 10 : 0
  const pctMl = result ? Math.round(result.ml_spam_probability * 1000) / 10 : 0

  return (
    <div className="app">
      <header className="header">
        <h1>Email spam detector</h1>
        <p className="lede">
          Fill in From, Subject, and body below, or upload a raw <code>.eml</code> file. The model
          uses the same fields (for uploads, parsed from the message).
        </p>
      </header>

      <section className="panel form-panel" aria-label="Manual message fields">
        <div className="field-block">
          <label className="label" htmlFor="from">
            From
          </label>
          <input
            id="from"
            type="text"
            className="input"
            autoComplete="off"
            placeholder="sender@example.com"
            value={fromAddr}
            onChange={(e) => setFromAddr(e.target.value)}
            disabled={loading}
          />
        </div>
        <div className="field-block">
          <label className="label" htmlFor="subject">
            Subject
          </label>
          <input
            id="subject"
            type="text"
            className="input"
            autoComplete="off"
            placeholder="Re: your order"
            value={subject}
            onChange={(e) => setSubject(e.target.value)}
            disabled={loading}
          />
        </div>
        <div className="field-block">
          <label className="label" htmlFor="body">
            Message body
          </label>
          <textarea
            id="body"
            className="textarea"
            rows={10}
            placeholder="Email body text…"
            value={body}
            onChange={(e) => setBody(e.target.value)}
            disabled={loading}
          />
        </div>
        <div className="actions">
          <button type="button" className="btn primary" onClick={runClassify} disabled={loading}>
            {loading ? 'Classifying…' : 'Classify text'}
          </button>
          <label className="btn secondary file-label">
            Upload .eml
            <input
              type="file"
              accept=".eml,.txt,message/rfc822"
              className="sr-only"
              disabled={loading}
              onChange={(e) => void onFile(e.target.files)}
            />
          </label>
        </div>
      </section>

      {error ? (
        <div className="banner error" role="alert">
          {error}
        </div>
      ) : null}

      {result ? (
        <section className={`result ${result.label}`} aria-live="polite">
          <div className="result-row">
            <span className="result-label">Verdict</span>
            <strong className="verdict">{result.label === 'spam' ? 'Spam' : 'Not spam'}</strong>
          </div>
          <div className="result-row">
            <span className="result-label">Combined score</span>
            <span>{pct}%</span>
          </div>
          <div className="meter" aria-hidden>
            <div
              className="meter-fill"
              style={{ width: `${Math.min(100, Math.max(0, pct))}%` }}
            />
          </div>
          <div className="result-row sub">
            <span className="result-label">Model-only (TF‑IDF)</span>
            <span>{pctMl}%</span>
          </div>
          <p className="hint">
            Verdict uses the combined score (model + light rules for modern promo cues like urgency
            lines and local prices). Spam if combined ≥ {Math.round(result.spam_threshold * 100)}%.
            Corpus is 2000s email—still imperfect. Model <code>{result.version}</code>.
          </p>
        </section>
      ) : null}

      <footer className="footer">
        API: <code>{apiBase}</code> (set <code>VITE_API_BASE_URL</code> to change)
      </footer>
    </div>
  )
}

export default App
