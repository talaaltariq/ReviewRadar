import { useState, useEffect, useRef } from 'react'

const BACKEND_URL = 'http://127.0.0.1:8000'

function App() {
  const [activeTab, setActiveTab] = useState('dashboard') // 'dashboard' | 'single' | 'batch'
  const [serverHealth, setServerHealth] = useState({ status: 'checking', message: '', models: null })
  
  // Dashboard Metrics State
  const [metricsData, setMetricsData] = useState(null)
  const [metricsLoading, setMetricsLoading] = useState(false)
  const [metricsError, setMetricsError] = useState(null)

  // Single Review State
  const [singleText, setSingleText] = useState('')
  const [singleModel, setSingleModel] = useState('minilm')
  const [singleResult, setSingleResult] = useState(null)
  const [singleLoading, setSingleLoading] = useState(false)
  const [singleError, setSingleError] = useState(null)

  // Batch CSV State
  const [batchFile, setBatchFile] = useState(null)
  const [batchModel, setBatchModel] = useState('minilm')
  const [batchLoading, setBatchLoading] = useState(false)
  const [batchError, setBatchError] = useState(null)
  const [batchResults, setBatchResults] = useState(null)

  // DOM Refs for automatic smooth scrolling to results
  const singleResultRef = useRef(null)
  const batchResultRef = useRef(null)

  // Smooth scroll to single review result when it arrives
  useEffect(() => {
    if (singleResult && singleResultRef.current) {
      singleResultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [singleResult])

  // Smooth scroll to batch CSV result when it arrives
  useEffect(() => {
    if (batchResults && batchResultRef.current) {
      batchResultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' })
    }
  }, [batchResults])

  // Check health and fetch metrics on startup
  useEffect(() => {
    fetchHealth()
    fetchMetrics()
  }, [])

  const fetchHealth = async () => {
    try {
      const res = await fetch(`${BACKEND_URL}/api/health`)
      if (res.ok) {
        const data = await res.json()
        setServerHealth({ status: 'online', message: data.message, models: data.models })
      } else {
        setServerHealth({ status: 'offline', message: `HTTP ${res.status}`, models: null })
      }
    } catch (err) {
      setServerHealth({ status: 'offline', message: 'Backend unreachable', models: null })
    }
  }

  const fetchMetrics = async () => {
    setMetricsLoading(true)
    setMetricsError(null)
    try {
      const res = await fetch(`${BACKEND_URL}/api/metrics`)
      if (res.ok) {
        const data = await res.json()
        if (data.available && data.data) {
          setMetricsData(data.data)
        } else {
          setMetricsData(null)
          setMetricsError(data.message || 'No metrics found in metrics.json.')
        }
      } else {
        setMetricsData(null)
        setMetricsError('Failed to retrieve metrics from backend.')
      }
    } catch (err) {
      setMetricsData(null)
      setMetricsError('Cannot reach backend metrics endpoint.')
    } finally {
      setMetricsLoading(false)
    }
  }

  // --- Single Review Handler ---
  const handleSingleAnalyze = async (e) => {
    if (e) e.preventDefault()
    if (!singleText.trim()) {
      setSingleError('Please enter some review text to analyze.')
      return
    }

    setSingleLoading(true)
    setSingleError(null)
    setSingleResult(null)

    try {
      const res = await fetch(`${BACKEND_URL}/api/predict`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          review_text: singleText.trim(),
          model: singleModel
        })
      })

      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || `Prediction failed (HTTP ${res.status})`)
      }

      setSingleResult(data)
    } catch (err) {
      setSingleError(err.message || 'Error communicating with backend.')
    } finally {
      setSingleLoading(false)
    }
  }

  // Quick sample insert for Single Review
  const insertSampleText = (text) => {
    setSingleText(text)
    setSingleResult(null)
    setSingleError(null)
  }

  // --- Batch CSV Handler ---
  const splitCSVRow = (line) => {
    const result = []
    let current = ''
    let inQuotes = false
    for (let i = 0; i < line.length; i++) {
      const char = line[i]
      if (char === '"') {
        if (inQuotes && line[i + 1] === '"') {
          current += '"'
          i++
        } else {
          inQuotes = !inQuotes
        }
      } else if (char === ',' && !inQuotes) {
        result.push(current.trim())
        current = ''
      } else {
        current += char
      }
    }
    result.push(current.trim())
    return result
  }

  const parseCSV = (csvText) => {
    const lines = csvText.split(/\r?\n/).filter(line => line.trim().length > 0)
    if (lines.length < 2) {
      throw new Error('CSV must contain a header row and at least one data row.')
    }

    // Parse header row
    const headers = splitCSVRow(lines[0]).map(h => h.replace(/^["']|["']$/g, '').toLowerCase())
    
    // Look for 'review_text' or 'text' column
    let colIndex = headers.indexOf('review_text')
    if (colIndex === -1) {
      colIndex = headers.indexOf('text')
    }
    if (colIndex === -1) {
      colIndex = headers.findIndex(h => h.includes('review') || h.includes('text'))
    }

    if (colIndex === -1) {
      throw new Error('CSV must contain a column named "review_text" or "text". Found headers: ' + headers.join(', '))
    }

    // Extract review texts safely handling empty cells, quotes, and commas inside quotes
    const reviews = []
    for (let i = 1; i < lines.length; i++) {
      const line = lines[i].trim()
      if (!line) continue

      const cols = splitCSVRow(line)
      if (cols[colIndex] !== undefined) {
        const val = cols[colIndex].replace(/^["']|["']$/g, '').trim()
        if (val) reviews.push(val)
      }
    }

    if (reviews.length === 0) {
      throw new Error('No valid non-empty review rows found under column "' + headers[colIndex] + '".')
    }

    return reviews
  }

  const handleBatchFileChange = (e) => {
    const file = e.target.files[0]
    setBatchFile(file || null)
    setBatchError(null)
    setBatchResults(null)
  }

  const handleBatchAnalyze = async (e) => {
    e.preventDefault()
    if (!batchFile) {
      setBatchError('Please select a CSV file first.')
      return
    }

    setBatchLoading(true)
    setBatchError(null)
    setBatchResults(null)

    try {
      const text = await batchFile.text()
      const reviews = parseCSV(text)

      const res = await fetch(`${BACKEND_URL}/api/predict-batch`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          reviews: reviews,
          model: batchModel
        })
      })

      const data = await res.json()
      if (!res.ok) {
        throw new Error(data.detail || `Batch request failed with status ${res.status}`)
      }

      setBatchResults(data)
    } catch (err) {
      setBatchError(err.message || 'Batch analysis failed.')
    } finally {
      setBatchLoading(false)
    }
  }

  // Generate and download a starter CSV template
  const downloadSampleCSV = () => {
    const sampleContent = `review_text\n"The build quality is outstanding and exceeded expectations!"\n"Average experience. The item works but delivery took two weeks."\n"Terrible product. Arrived broken and seller refused refund."\n"Fantastic value for money, highly recommended to all."\n"Neutral feeling. Nothing special but gets the job done."\n`
    const blob = new Blob([sampleContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'sample_reviews.csv')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Download enriched results CSV
  const downloadBatchResults = () => {
    if (!batchResults || batchResults.length === 0) return

    let csvContent = 'row_id,review_text,predicted_sentiment,confidence,model_used\n'
    batchResults.forEach((row, idx) => {
      const escapedText = `"${(row.review_text || '').replace(/"/g, '""')}"`
      const sentiment = row.sentiment || 'Unknown'
      const confidence = row.confidence !== undefined ? row.confidence : ''
      const model = `"${(row.model_used || '').replace(/"/g, '""')}"`
      csvContent += `${idx + 1},${escapedText},${sentiment},${confidence},${model}\n`
    })

    const blob = new Blob([csvContent], { type: 'text/csv;charset=utf-8;' })
    const url = URL.createObjectURL(blob)
    const link = document.createElement('a')
    link.href = url
    link.setAttribute('download', 'reviewradar_predictions.csv')
    document.body.appendChild(link)
    link.click()
    document.body.removeChild(link)
  }

  // Helpers for batch summary counts
  const computeBatchCounts = () => {
    if (!batchResults) return { positive: 0, neutral: 0, negative: 0, total: 0 }
    let pos = 0, neu = 0, neg = 0
    batchResults.forEach(r => {
      const s = (r.sentiment || '').toLowerCase()
      if (s === 'positive') pos++
      else if (s === 'neutral') neu++
      else if (s === 'negative') neg++
    })
    return { positive: pos, neutral: neu, negative: neg, total: batchResults.length }
  }

  // Helper to extract accuracy & macro F1 from metricsData structure
  const extractModelMetrics = (data, modelKey) => {
    if (!data) return null
    if (data.evaluation_results && data.evaluation_results[modelKey]) {
      const m = data.evaluation_results[modelKey]
      return {
        name: m.Model || modelKey,
        accuracy: m.Accuracy !== undefined ? m.Accuracy : m.accuracy,
        macroF1: m['Macro F1'] !== undefined ? m['Macro F1'] : m.macro_f1 || m['macro f1']
      }
    }
    if (data[modelKey]) {
      const m = data[modelKey]
      return {
        name: m.Model || m.name || modelKey,
        accuracy: m.Accuracy !== undefined ? m.Accuracy : m.accuracy,
        macroF1: m['Macro F1'] !== undefined ? m['Macro F1'] : m.macro_f1 || m['macro f1']
      }
    }
    return null
  }

  const m1 = metricsData ? extractModelMetrics(metricsData, 'model_1_tfidf_lr') : null
  const m2 = metricsData ? extractModelMetrics(metricsData, 'model_2_minilm_lr') : null

  // Extract saved sentiment distribution metadata if available
  const extractSavedDistribution = () => {
    if (!metricsData || !metricsData.sentiment_distribution) return null
    const dist = metricsData.sentiment_distribution.combined || metricsData.sentiment_distribution.test || metricsData.sentiment_distribution
    if (typeof dist !== 'object') return null
    
    const pos = dist.Positive || dist.positive || 0
    const neu = dist.Neutral || dist.neutral || 0
    const neg = dist.Negative || dist.negative || 0
    const total = pos + neu + neg
    if (total === 0) return null

    return {
      positive: pos,
      neutral: neu,
      negative: neg,
      total: total,
      posPct: ((pos / total) * 100).toFixed(1),
      neuPct: ((neu / total) * 100).toFixed(1),
      negPct: ((neg / total) * 100).toFixed(1)
    }
  }

  const savedDist = extractSavedDistribution()
  const batchCounts = computeBatchCounts()

  return (
    <div>
      {/* Floating Contained Wide-Pill Navbar */}
      <header className="ui-nav-wide-pill">
        <div className="ui-brand">
          <span className="ui-brand-mark"></span>
          <span>ReviewRadar</span>
        </div>

        <nav className="ui-nav-tabs" aria-label="Workspace Views">
          <button
            type="button"
            className={`ui-nav-tab ${activeTab === 'dashboard' ? 'active' : ''}`}
            onClick={() => setActiveTab('dashboard')}
          >
            Overview
          </button>
          <button
            type="button"
            className={`ui-nav-tab ${activeTab === 'single' ? 'active' : ''}`}
            onClick={() => setActiveTab('single')}
          >
            Single Review
          </button>
          <button
            type="button"
            className={`ui-nav-tab ${activeTab === 'batch' ? 'active' : ''}`}
            onClick={() => setActiveTab('batch')}
          >
            Batch CSV
          </button>
        </nav>
      </header>

      <main className="ui-container">
        {/* Compact Editorial Hero */}
        <section className="ui-hero-compact">
          <h1 className="ui-display-sans">
            Customer sentiment, <em className="ui-serif-italic">distilled</em>.
          </h1>
        </section>

        {/* ================= TAB 1: DASHBOARD ================= */}
        {activeTab === 'dashboard' && (
          <section>
            <div className="ui-section-header-split">
              <div>
                <span className="ui-step-num">EVALUATION METRICS</span>
                <h2 className="ui-h2-mixed">
                  Model Performance <em className="ui-serif-italic">Comparison</em>
                </h2>
              </div>
              <div style={{ display: 'flex', alignItems: 'center', gap: '12px' }}>
                <span className="ui-mono-stat" style={{ color: 'var(--text-muted)' }}>
                  Source: models/metrics.json
                </span>
                <button
                  type="button"
                  className="ui-btn ui-btn-outline ui-btn-sm"
                  onClick={fetchMetrics}
                  disabled={metricsLoading}
                >
                  {metricsLoading ? 'Refreshing...' : 'Refresh'}
                </button>
              </div>
            </div>

            {metricsError && !metricsLoading && (
              <div className="ui-bento-cell" style={{ marginBottom: 'var(--space-5)' }}>
                <div>
                  <span className="ui-step-num">STATUS</span>
                  <h3 className="ui-bento-title">Metrics Artifact Not Yet Loaded</h3>
                  <p className="ui-bento-desc">{metricsError}</p>
                </div>
                <div className="ui-notice">
                  Export <code>metrics.json</code> from your Google Colab evaluation notebook into the local{' '}
                  <code>models/</code> folder, then click <strong>Refresh</strong> to display recorded test scores.
                </div>
              </div>
            )}

            <div className="ui-bento-grid-2x2">
              {/* Model 1 Bento Cell */}
              <article className="ui-bento-cell">
                <div>
                  <span className="ui-step-num">001 · BASELINE PIPELINE</span>
                  <h3 className="ui-bento-title">TF-IDF + Logistic Regression</h3>
                </div>
                <div className="ui-bento-widget">
                  <div className="ui-metric-pair">
                    <div className="ui-metric-stat-box">
                      <span className="ui-metric-stat-label">Accuracy</span>
                      <span className="ui-metric-stat-value">
                        {m1 && m1.accuracy !== undefined ? `${(m1.accuracy * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                    <div className="ui-metric-stat-box">
                      <span className="ui-metric-stat-label">Macro F1</span>
                      <span className="ui-metric-stat-value">
                        {m1 && m1.macroF1 !== undefined ? m1.macroF1.toFixed(3) : '—'}
                      </span>
                    </div>
                  </div>
                </div>
              </article>

              {/* Model 2 Bento Cell */}
              <article className="ui-bento-cell">
                <div>
                  <span className="ui-step-num">002 · TRANSFORMER EMBEDDINGS</span>
                  <h3 className="ui-bento-title">all-MiniLM-L6-v2 + Logistic Regression</h3>
                </div>
                <div className="ui-bento-widget">
                  <div className="ui-metric-pair">
                    <div className="ui-metric-stat-box">
                      <span className="ui-metric-stat-label">Accuracy</span>
                      <span className="ui-metric-stat-value">
                        {m2 && m2.accuracy !== undefined ? `${(m2.accuracy * 100).toFixed(1)}%` : '—'}
                      </span>
                    </div>
                    <div className="ui-metric-stat-box">
                      <span className="ui-metric-stat-label">Macro F1</span>
                      <span className="ui-metric-stat-value">
                        {m2 && m2.macroF1 !== undefined ? m2.macroF1.toFixed(3) : '—'}
                      </span>
                    </div>
                  </div>
                </div>
              </article>
            </div>

            {/* Sentiment Distribution Bento Card (if saved metadata exists) */}
            {savedDist && (
              <article className="ui-bento-cell" style={{ marginTop: 'var(--space-5)' }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap' }}>
                  <div>
                    <span className="ui-step-num">003 · DATASET METADATA</span>
                    <h3 className="ui-bento-title">Sentiment Class Distribution</h3>
                  </div>
                  <span className="ui-mono-stat" style={{ color: 'var(--text-secondary)' }}>
                    {savedDist.total.toLocaleString()} usable reviews
                  </span>
                </div>

                <div className="ui-bento-widget">
                  <div className="ui-bar-stack">
                    <div className="ui-bar-item">
                      <div className="ui-bar-meta">
                        <span className="ui-bar-label">Positive</span>
                        <span className="ui-bar-count">{savedDist.positive.toLocaleString()} ({savedDist.posPct}%)</span>
                      </div>
                      <div className="ui-bar-track">
                        <div className="ui-bar-fill positive" style={{ width: `${savedDist.posPct}%` }}></div>
                      </div>
                    </div>

                    <div className="ui-bar-item">
                      <div className="ui-bar-meta">
                        <span className="ui-bar-label">Neutral</span>
                        <span className="ui-bar-count">{savedDist.neutral.toLocaleString()} ({savedDist.neuPct}%)</span>
                      </div>
                      <div className="ui-bar-track">
                        <div className="ui-bar-fill neutral" style={{ width: `${savedDist.neuPct}%` }}></div>
                      </div>
                    </div>

                    <div className="ui-bar-item">
                      <div className="ui-bar-meta">
                        <span className="ui-bar-label">Negative</span>
                        <span className="ui-bar-count">{savedDist.negative.toLocaleString()} ({savedDist.negPct}%)</span>
                      </div>
                      <div className="ui-bar-track">
                        <div className="ui-bar-fill negative" style={{ width: `${savedDist.negPct}%` }}></div>
                      </div>
                    </div>
                  </div>
                </div>
              </article>
            )}
          </section>
        )}

        {/* ================= TAB 2: SINGLE REVIEW ================= */}
        {activeTab === 'single' && (
          <section>
            <div className="ui-bento-cell">
              <div>
                <span className="ui-step-num">LIVE INFERENCE</span>
                <h2 className="ui-h2-mixed">
                  Analyze a <em className="ui-serif-italic">Single Review</em>
                </h2>
              </div>

              <div className="ui-bento-widget">
                <form onSubmit={handleSingleAnalyze}>
                  <div className="ui-field-group">
                    <label htmlFor="singleModelSelect" className="ui-label">
                      <span>Model Architecture</span>
                    </label>
                    <select
                      id="singleModelSelect"
                      className="ui-select"
                      value={singleModel}
                      onChange={(e) => setSingleModel(e.target.value)}
                    >
                      <option value="minilm">all-MiniLM-L6-v2 Embeddings + Logistic Regression (Main)</option>
                      <option value="tfidf">TF-IDF + Logistic Regression (Baseline)</option>
                    </select>
                  </div>

                  <div className="ui-field-group">
                    <label htmlFor="singleReviewText" className="ui-label">
                      <span>Customer Review Text</span>
                    </label>
                    <textarea
                      id="singleReviewText"
                      className="ui-textarea"
                      rows="4"
                      placeholder="Paste or type a product or service review..."
                      value={singleText}
                      onChange={(e) => setSingleText(e.target.value)}
                    />
                    <div style={{ display: 'flex', alignItems: 'center', flexWrap: 'wrap', gap: '6px', marginTop: '6px' }}>
                      <span className="ui-mono-stat" style={{ color: 'var(--text-muted)', fontSize: '0.75rem', marginRight: '4px' }}>
                        Samples:
                      </span>
                      <button
                        type="button"
                        className="ui-token-pill"
                        onClick={() => insertSampleText('Amazing quality! The packaging was great and the product works flawlessly.')}
                      >
                        Positive
                      </button>
                      <button
                        type="button"
                        className="ui-token-pill"
                        onClick={() => insertSampleText('The item arrived on time. Functions okay, nothing extraordinary.')}
                      >
                        Neutral
                      </button>
                      <button
                        type="button"
                        className="ui-token-pill"
                        onClick={() => insertSampleText('Terrible experience. The item was broken on arrival and support was completely useless.')}
                      >
                        Negative
                      </button>
                    </div>
                  </div>

                  <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', flexWrap: 'wrap', gap: '12px' }}>
                    <button
                      type="submit"
                      className="ui-btn ui-btn-obsidian"
                      disabled={singleLoading || !singleText.trim()}
                    >
                      <span>{singleLoading ? 'Analyzing...' : 'Analyze Review'}</span>
                      <span className="ui-arrow">→</span>
                    </button>
                  </div>
                </form>

                {singleError && (
                  <div className="ui-notice error" style={{ marginTop: 'var(--space-5)' }}>
                    <strong>Error: </strong> {singleError}
                  </div>
                )}
              </div>

              {singleResult && (() => {
                const sKey = (singleResult.sentiment || 'Neutral').toLowerCase()
                const confDisplay = singleResult.confidence_percentage || (singleResult.confidence !== undefined ? `${(singleResult.confidence * 100).toFixed(1)}%` : 'N/A')

                return (
                  <div
                    ref={singleResultRef}
                    key={`${singleResult.sentiment}-${singleResult.confidence}-${singleResult.model_used}`}
                    className="ui-result-popup-card"
                  >
                    {/* Large Centered Minimal Sentiment Graphic with Gesture Badge */}
                    <div className={`ui-sentiment-stage ${sKey}`} aria-hidden="true">
                      {sKey === 'positive' && (
                        <>
                          {/* Minimal Happy Face */}
                          <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="10" />
                            <path d="M8 14s1.5 2.5 4 2.5 4-2.5 4-2.5" />
                            <line x1="9" y1="9.5" x2="9.01" y2="9.5" strokeWidth="2.5" />
                            <line x1="15" y1="9.5" x2="15.01" y2="9.5" strokeWidth="2.5" />
                          </svg>
                          {/* Floating Thumbs Up Badge */}
                          <span className="ui-sentiment-gesture-badge">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M7 10v12" />
                              <path d="M15 5.88 14 10h5.83a2 2 0 0 1 1.92 2.56l-2.33 8A2 2 0 0 1 17.5 22H4a2 2 0 0 1-2-2v-8a2 2 0 0 1 2-2h2.76a2 2 0 0 0 1.79-1.11L12 2a3.13 3.13 0 0 1 3 3.88Z" />
                            </svg>
                          </span>
                        </>
                      )}

                      {sKey === 'neutral' && (
                        <>
                          {/* Minimal Neutral Face */}
                          <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="10" />
                            <line x1="8.5" y1="15" x2="15.5" y2="15" />
                            <line x1="9" y1="9.5" x2="9.01" y2="9.5" strokeWidth="2.5" />
                            <line x1="15" y1="9.5" x2="15.01" y2="9.5" strokeWidth="2.5" />
                          </svg>
                          {/* Floating Balanced / Minus Gesture Badge */}
                          <span className="ui-sentiment-gesture-badge">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2.2" strokeLinecap="round" strokeLinejoin="round">
                              <line x1="5" y1="9" x2="19" y2="9" />
                              <line x1="5" y1="15" x2="19" y2="15" />
                            </svg>
                          </span>
                        </>
                      )}

                      {sKey === 'negative' && (
                        <>
                          {/* Minimal Frown Face */}
                          <svg width="56" height="56" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.75" strokeLinecap="round" strokeLinejoin="round">
                            <circle cx="12" cy="12" r="10" />
                            <path d="M16 16.5s-1.5-2.5-4-2.5-4 2.5-4 2.5" />
                            <line x1="9" y1="9.5" x2="9.01" y2="9.5" strokeWidth="2.5" />
                            <line x1="15" y1="9.5" x2="15.01" y2="9.5" strokeWidth="2.5" />
                          </svg>
                          {/* Floating Thumbs Down Badge */}
                          <span className="ui-sentiment-gesture-badge">
                            <svg width="18" height="18" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round">
                              <path d="M17 14V2" />
                              <path d="M9 18.12 10 14H4.17a2 2 0 0 1-1.92-2.56l2.33-8A2 2 0 0 1 6.5 2H20a2 2 0 0 1 2 2v8a2 2 0 0 1-2 2h-2.76a2 2 0 0 0-1.79 1.11L12 22a3.13 3.13 0 0 1-3-3.88Z" />
                            </svg>
                          </span>
                        </>
                      )}
                    </div>

                    <span className="ui-step-num" style={{ marginBottom: '4px' }}>
                      {singleResult.model_used}
                    </span>

                    <div className="ui-result-popup-metrics">
                      <div className="ui-metric-stat-box" style={{ alignItems: 'center' }}>
                        <span className="ui-metric-stat-label">Sentiment</span>
                        <span className="ui-metric-stat-value" style={{ fontSize: '1.65rem' }}>
                          {singleResult.sentiment}
                        </span>
                      </div>

                      <div className="ui-result-popup-divider" />

                      <div className="ui-metric-stat-box" style={{ alignItems: 'center' }}>
                        <span className="ui-metric-stat-label">Confidence</span>
                        <span className="ui-metric-stat-value" style={{ fontSize: '1.65rem' }}>
                          {confDisplay}
                        </span>
                      </div>
                    </div>
                  </div>
                )
              })()}
            </div>
          </section>
        )}

        {/* ================= TAB 3: BATCH CSV ================= */}
        {activeTab === 'batch' && (
          <section>
            <div className="ui-bento-cell">
              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-end', flexWrap: 'wrap', gap: '12px' }}>
                <div>
                  <span className="ui-step-num">BULK PROCESSING</span>
                  <h2 className="ui-h2-mixed">
                    Batch CSV <em className="ui-serif-italic">Evaluation</em>
                  </h2>
                </div>
                <button type="button" className="ui-btn ui-btn-outline ui-btn-sm" onClick={downloadSampleCSV}>
                  Download Sample CSV
                </button>
              </div>

              <div className="ui-bento-widget">
                <form onSubmit={handleBatchAnalyze}>
                  <div className="ui-field-group">
                    <label htmlFor="batchModelSelect" className="ui-label">
                      <span>Model Architecture</span>
                    </label>
                    <select
                      id="batchModelSelect"
                      className="ui-select"
                      value={batchModel}
                      onChange={(e) => setBatchModel(e.target.value)}
                    >
                      <option value="minilm">all-MiniLM-L6-v2 Embeddings + Logistic Regression (Main)</option>
                      <option value="tfidf">TF-IDF + Logistic Regression (Baseline)</option>
                    </select>
                  </div>

                  <div className="ui-field-group">
                    <label htmlFor="batchFileInput" className="ui-label">
                      <span>CSV Dataset File</span>
                      <span className="ui-mono-stat" style={{ color: 'var(--text-muted)', fontSize: '0.725rem' }}>
                        Requires review_text or text column
                      </span>
                    </label>
                    <div className="ui-file-drop">
                      <input
                        id="batchFileInput"
                        type="file"
                        accept=".csv"
                        onChange={handleBatchFileChange}
                        style={{ fontSize: '0.875rem', color: 'var(--text-secondary)' }}
                      />
                    </div>
                  </div>

                  <button
                    type="submit"
                    className="ui-btn ui-btn-obsidian"
                    disabled={batchLoading || !batchFile}
                  >
                    <span>{batchLoading ? 'Processing Batch...' : 'Upload & Classify'}</span>
                    <span className="ui-arrow">→</span>
                  </button>
                </form>

                {batchError && (
                  <div className="ui-notice error" style={{ marginTop: 'var(--space-5)' }}>
                    <strong>Error: </strong> {batchError}
                  </div>
                )}
              </div>

              {batchResults && (
                <div ref={batchResultRef} style={{ display: 'flex', flexDirection: 'column', gap: 'var(--space-5)' }}>
                  {/* Batch Summary & Distribution Widget */}
                  <div className="ui-bento-widget">
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: '12px', marginBottom: 'var(--space-5)' }}>
                      <div>
                        <span className="ui-step-num" style={{ marginBottom: '2px' }}>BATCH DISTRIBUTION</span>
                        <h3 className="ui-bento-title">{batchResults.length} Reviews Classified</h3>
                      </div>
                      <button type="button" className="ui-btn ui-btn-outline ui-btn-sm" onClick={downloadBatchResults}>
                        Export Predictions CSV ↓
                      </button>
                    </div>

                    <div className="ui-bar-stack">
                      <div className="ui-bar-item">
                        <div className="ui-bar-meta">
                          <span className="ui-bar-label">Positive</span>
                          <span className="ui-bar-count">
                            {batchCounts.positive} ({batchCounts.total > 0 ? ((batchCounts.positive / batchCounts.total) * 100).toFixed(1) : 0}%)
                          </span>
                        </div>
                        <div className="ui-bar-track">
                          <div
                            className="ui-bar-fill positive"
                            style={{ width: `${batchCounts.total > 0 ? (batchCounts.positive / batchCounts.total) * 100 : 0}%` }}
                          ></div>
                        </div>
                      </div>

                      <div className="ui-bar-item">
                        <div className="ui-bar-meta">
                          <span className="ui-bar-label">Neutral</span>
                          <span className="ui-bar-count">
                            {batchCounts.neutral} ({batchCounts.total > 0 ? ((batchCounts.neutral / batchCounts.total) * 100).toFixed(1) : 0}%)
                          </span>
                        </div>
                        <div className="ui-bar-track">
                          <div
                            className="ui-bar-fill neutral"
                            style={{ width: `${batchCounts.total > 0 ? (batchCounts.neutral / batchCounts.total) * 100 : 0}%` }}
                          ></div>
                        </div>
                      </div>

                      <div className="ui-bar-item">
                        <div className="ui-bar-meta">
                          <span className="ui-bar-label">Negative</span>
                          <span className="ui-bar-count">
                            {batchCounts.negative} ({batchCounts.total > 0 ? ((batchCounts.negative / batchCounts.total) * 100).toFixed(1) : 0}%)
                          </span>
                        </div>
                        <div className="ui-bar-track">
                          <div
                            className="ui-bar-fill negative"
                            style={{ width: `${batchCounts.total > 0 ? (batchCounts.negative / batchCounts.total) * 100 : 0}%` }}
                          ></div>
                        </div>
                      </div>
                    </div>
                  </div>

                  {/* Results Table */}
                  <div className="ui-table-wrap">
                    <table className="ui-table">
                      <thead>
                        <tr>
                          <th style={{ width: '56px' }}>#</th>
                          <th>Review Text</th>
                          <th style={{ width: '130px' }}>Sentiment</th>
                          <th style={{ width: '110px' }}>Confidence</th>
                        </tr>
                      </thead>
                      <tbody>
                        {batchResults.slice(0, 50).map((row, idx) => (
                          <tr key={idx}>
                            <td className="ui-mono-stat" style={{ color: 'var(--text-muted)' }}>
                              {String(idx + 1).padStart(2, '0')}
                            </td>
                            <td className="ui-text-truncate" title={row.review_text}>
                              {row.review_text}
                            </td>
                            <td>
                              <span className={`ui-sentiment-pill ${row.sentiment.toLowerCase()}`}>
                                {row.sentiment}
                              </span>
                            </td>
                            <td className="ui-mono-stat">
                              {row.confidence_percentage || (row.confidence !== undefined ? `${(row.confidence * 100).toFixed(1)}%` : 'N/A')}
                            </td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>

                  {batchResults.length > 50 && (
                    <span className="ui-micro-note" style={{ textAlign: 'center', marginTop: 0 }}>
                      Showing first 50 of {batchResults.length} records. Export CSV to inspect all rows.
                    </span>
                  )}
                </div>
              )}
            </div>
          </section>
        )}
      </main>
    </div>
  )
}

export default App
