import { useState } from 'react';
import axios from 'axios';
import {
  UploadCloud,
  FileText,
  ChevronDown,
  ChevronUp,
  CheckCircle2,
  XCircle,
  AlertCircle,
  Loader2,
  Sparkles,
  Trophy,
  Star,
  FileBox,
  Type
} from 'lucide-react';

interface SkillGapReport {
  match_rate: number;
  matched: string[];
  missing: string[];
  bonus: string[];
}

interface MatchData {
  overall_match_percentage: number;
  semantic_confidence: number;
  coherence_gap: number;
  density_penalty_applied: boolean;
  skill_gap_report: SkillGapReport;
  dealbreakers: string[];
}

interface CandidateResult {
  filename: string;
  match_data: MatchData;
}

interface FailedFile {
  filename: string;
  error: string;
}

interface ApiResponse {
  jd_processed: boolean;
  total_candidates: number;
  ranked_results: CandidateResult[];
  failed_files: FailedFile[];
}

const ScoreRing = ({ score: rawScore }: { score: number }) => {
  const score = rawScore ?? 0;
  const r = 28;
  const circ = 2 * Math.PI * r;
  const color = score >= 80 ? '#10b981' : score >= 50 ? '#f59e0b' : '#ef4444';
  return (
    <svg width="72" height="72" viewBox="0 0 72 72" style={{ flexShrink: 0 }}>
      <circle cx="36" cy="36" r={r} fill="none" stroke="#e8e3db" strokeWidth="4" />
      <circle
        cx="36" cy="36" r={r} fill="none" stroke={color} strokeWidth="4"
        strokeDasharray={circ} strokeDashoffset={circ * (1 - score / 100)}
        strokeLinecap="round" transform="rotate(-90 36 36)"
        style={{ transition: 'stroke-dashoffset 0.8s cubic-bezier(0.4,0,0.2,1)' }}
      />
      <text x="36" y="40" textAnchor="middle" fontSize="13" fontWeight="700" fill={color} fontFamily="'DM Mono', monospace">
        {score.toFixed(0)}%
      </text>
    </svg>
  );
};

const getRankBadge = (idx: number) => {
  if (idx === 0) return { icon: <Trophy style={{ width: 12, height: 12 }} />, label: 'Top Match', bg: '#fef3c7', border: '#fde68a', color: '#92400e' };
  if (idx === 1) return { icon: <Star style={{ width: 12, height: 12 }} />, label: 'Strong Fit', bg: '#f1f5f9', border: '#e2e8f0', color: '#475569' };
  return null;
};

export default function App() {
  const [jdText, setJdText] = useState('');
  const [inputType, setInputType] = useState<'pdf' | 'text'>('pdf');
  const [resumeText, setResumeText] = useState('');
  const [files, setFiles] = useState<File[]>([]);
  const [isDragging, setIsDragging] = useState(false);
  const [isLoading, setIsLoading] = useState(false);
  const [results, setResults] = useState<CandidateResult[] | null>(null);
  const [failedFiles, setFailedFiles] = useState<FailedFile[] | null>(null);
  const [expandedCard, setExpandedCard] = useState<string | null>(null);
  const [rejectionError, setRejectionError] = useState<string | null>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files) setFiles(Array.from(e.target.files));
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    setIsDragging(false);
    const dropped = Array.from(e.dataTransfer.files).filter(f => f.type === 'application/pdf');
    if (dropped.length) setFiles(dropped);
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!jdText.trim()) return;
    if (inputType === 'pdf' && files.length === 0) return;
    if (inputType === 'text' && !resumeText.trim()) return;

    setIsLoading(true);
    setResults(null);
    setFailedFiles(null);
    setExpandedCard(null);
    setRejectionError(null);

    try {
      if (inputType === 'pdf') {
        const formData = new FormData();
        formData.append('jd_text', jdText);
        files.forEach(f => formData.append('resumes', f));
        const res = await axios.post<ApiResponse>('http://localhost:8000/api/match-bulk', formData, {
          headers: { 'Content-Type': 'multipart/form-data' },
        });
        setResults(res.data.ranked_results);
        if (res.data.failed_files?.length > 0) setFailedFiles(res.data.failed_files);
      } else {
        const res = await axios.post<any>('http://localhost:8000/api/match', {
          jd_text: jdText,
          resume_text: resumeText,
        });
        const data = res.data;
        if (data?.error) {
          setRejectionError(data.message ?? 'Document too short or invalid. Please paste more content.');
          setIsLoading(false);
          return;
        }
        setResults([{ filename: 'Raw Text Input', match_data: data as MatchData }]);
      }
    } catch {
      alert('Backend unreachable. Make sure your FastAPI server is running.');
    } finally {
      setIsLoading(false);
    }
  };

  const toggleExpand = (fn: string) => setExpandedCard(expandedCard === fn ? null : fn);

  const canSubmit = jdText.trim().length > 0 && !isLoading &&
    (inputType === 'pdf' ? files.length > 0 : resumeText.trim().length > 0);

  const textareaStyle = (active: boolean): React.CSSProperties => ({
    width: '100%', resize: 'none', borderRadius: 10, border: '1px solid',
    borderColor: active ? 'rgba(201,169,110,0.5)' : 'rgba(0,0,0,0.1)',
    background: active ? 'rgba(201,169,110,0.03)' : '#fafaf9',
    padding: '12px 14px', fontSize: 13, color: '#1c1917', lineHeight: 1.6,
    fontFamily: "'DM Sans',sans-serif", transition: 'border-color 0.2s,background 0.2s', outline: 'none'
  });

  return (
    <div style={{ minHeight: '100vh', background: '#f5f0e8', fontFamily: "'DM Sans', sans-serif" }}>
      <style>{`
        @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@300;400;500;600;700&family=DM+Mono:wght@400;500&display=swap');
        * { box-sizing: border-box; }
        ::selection { background: rgba(180,155,120,0.3); }
        .fade-in { animation: fadeUp 0.35s cubic-bezier(0.4,0,0.2,1) both; }
        @keyframes fadeUp { from { opacity:0; transform:translateY(8px); } to { opacity:1; transform:translateY(0); } }
        .spin { animation: spin 1s linear infinite; }
        @keyframes spin { to { transform: rotate(360deg); } }
        .tag-matched { background:#ecfdf5; border:1px solid #a7f3d0; color:#047857; }
        .tag-missing  { background:#fff1f2; border:1px solid #fecdd3; color:#be123c; }
        .tag-bonus    { background:#eff6ff; border:1px solid #bfdbfe; color:#1d4ed8; }
        .progress-track { height:3px; border-radius:9999px; background:#e8e3db; overflow:hidden; }
        .result-card:hover { border-color: rgba(0,0,0,0.14) !important; }
        .toggle-btn { flex:1; display:flex; align-items:center; justify-content:center; gap:6px; padding:8px 0; font-size:13px; font-weight:500; border-radius:8px; border:none; cursor:pointer; transition:all 0.18s; font-family:'DM Sans',sans-serif; }
      `}</style>

      {/* Header */}
      <header style={{ background: 'rgba(245,240,232,0.9)', borderBottom: '1px solid rgba(0,0,0,0.08)', backdropFilter: 'blur(12px)', position: 'sticky', top: 0, zIndex: 20 }}>
        <div style={{ maxWidth: 1200, margin: '0 auto', padding: '0 24px', height: 56, display: 'flex', alignItems: 'center', gap: 10 }}>
          <div style={{ width: 28, height: 28, borderRadius: 8, background: 'linear-gradient(135deg,#c9a96e,#8b6d42)', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
            <Sparkles style={{ width: 14, height: 14, color: '#fff' }} />
          </div>
          <span style={{ fontWeight: 600, fontSize: 15, color: '#1c1917', letterSpacing: '-0.01em' }}>ATS Semantic Engine</span>
          <span style={{ fontSize: 12, color: '#a8a29e', marginLeft: 4, fontFamily: "'DM Mono',monospace" }}>v1.0.0</span>
        </div>
      </header>

      <main style={{ maxWidth: 1200, margin: '0 auto', padding: '32px 24px' }}>
        <div style={{ display: 'grid', gridTemplateColumns: '360px 1fr', gap: 24, alignItems: 'start' }}>

          {/* ── Left Panel ── */}
          <div style={{ background: '#fff', borderRadius: 16, padding: 24, border: '1px solid rgba(0,0,0,0.07)', boxShadow: '0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.06)', position: 'sticky', top: 72 }}>
            <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 20 }}>
              <FileText style={{ width: 16, height: 16, color: '#c9a96e' }} />
              <span style={{ fontWeight: 600, fontSize: 14, color: '#1c1917' }}>Target Parameters</span>
            </div>

            <form onSubmit={handleSubmit} style={{ display: 'flex', flexDirection: 'column', gap: 18 }}>

              {/* JD */}
              <div style={{ position: 'relative' }}>
                <label style={{ fontSize: 11, fontWeight: 600, color: '#78716c', display: 'block', marginBottom: 6, letterSpacing: '0.06em', textTransform: 'uppercase' as const }}>
                  Job Description
                </label>
                <textarea rows={8} value={jdText} onChange={e => setJdText(e.target.value)}
                  disabled={isLoading} placeholder="Paste job description here…" style={textareaStyle(!!jdText)} />
                {jdText && (
                  <div style={{ position: 'absolute', bottom: 10, right: 12, fontSize: 11, color: '#a8a29e', fontFamily: "'DM Mono',monospace" }}>
                    {jdText.split(/\s+/).filter(Boolean).length}w
                  </div>
                )}
              </div>

              {/* Mode Toggle */}
              <div>
                <label style={{ fontSize: 11, fontWeight: 600, color: '#78716c', display: 'block', marginBottom: 6, letterSpacing: '0.06em', textTransform: 'uppercase' as const }}>
                  Payload Type
                </label>
                <div style={{ display: 'flex', background: '#f5f0e8', borderRadius: 10, padding: 4, gap: 4 }}>
                  <button type="button" className="toggle-btn" onClick={() => setInputType('pdf')}
                    style={{ background: inputType === 'pdf' ? '#fff' : 'transparent', color: inputType === 'pdf' ? '#1c1917' : '#78716c', boxShadow: inputType === 'pdf' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none' }}>
                    <FileBox style={{ width: 14, height: 14 }} /> Bulk PDFs
                  </button>
                  <button type="button" className="toggle-btn" onClick={() => setInputType('text')}
                    style={{ background: inputType === 'text' ? '#fff' : 'transparent', color: inputType === 'text' ? '#1c1917' : '#78716c', boxShadow: inputType === 'text' ? '0 1px 3px rgba(0,0,0,0.1)' : 'none' }}>
                    <Type style={{ width: 14, height: 14 }} /> Raw Text
                  </button>
                </div>
              </div>

              {/* Dynamic Input */}
              {inputType === 'pdf' ? (
                <div>
                  <label style={{ fontSize: 11, fontWeight: 600, color: '#78716c', display: 'block', marginBottom: 6, letterSpacing: '0.06em', textTransform: 'uppercase' as const }}>
                    Résumés (PDF)
                  </label>
                  <div style={{ position: 'relative', borderRadius: 10, border: '2px dashed', transition: 'all 0.2s',
                    borderColor: isDragging ? '#c9a96e' : files.length ? 'rgba(201,169,110,0.5)' : 'rgba(0,0,0,0.12)',
                    background: isDragging ? 'rgba(201,169,110,0.06)' : files.length ? 'rgba(201,169,110,0.04)' : '#fafaf9' }}
                    onDragOver={e => { e.preventDefault(); setIsDragging(true); }}
                    onDragLeave={() => setIsDragging(false)} onDrop={handleDrop}>
                    <input type="file" multiple accept=".pdf" onChange={handleFileChange} disabled={isLoading}
                      style={{ position: 'absolute', inset: 0, opacity: 0, cursor: 'pointer', zIndex: 2, width: '100%', height: '100%' }} />
                    <div style={{ padding: '18px 16px', display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 6 }}>
                      <div style={{ width: 36, height: 36, borderRadius: 10, background: files.length ? 'rgba(201,169,110,0.12)' : '#f0ede8', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                        <UploadCloud style={{ width: 18, height: 18, color: files.length ? '#c9a96e' : '#a8a29e' }} />
                      </div>
                      <span style={{ fontSize: 13, fontWeight: 500, color: files.length ? '#92400e' : '#57534e' }}>
                        {files.length ? `${files.length} file${files.length > 1 ? 's' : ''} ready` : 'Drop PDFs or click to browse'}
                      </span>
                      {files.length > 0 && (
                        <div style={{ display: 'flex', flexDirection: 'column', gap: 3, width: '100%', marginTop: 4 }}>
                          {files.slice(0, 3).map((f, i) => (
                            <div key={i} style={{ fontSize: 11, color: '#78716c', background: 'rgba(201,169,110,0.08)', borderRadius: 6, padding: '3px 8px', fontFamily: "'DM Mono',monospace", overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                              {f.name}
                            </div>
                          ))}
                          {files.length > 3 && <div style={{ fontSize: 11, color: '#a8a29e', textAlign: 'center' }}>+{files.length - 3} more</div>}
                        </div>
                      )}
                    </div>
                  </div>
                </div>
              ) : (
                <div style={{ position: 'relative' }}>
                  <label style={{ fontSize: 11, fontWeight: 600, color: '#78716c', display: 'block', marginBottom: 6, letterSpacing: '0.06em', textTransform: 'uppercase' as const }}>
                    Raw Résumé Text
                  </label>
                  <textarea rows={8} value={resumeText} onChange={e => setResumeText(e.target.value)}
                    disabled={isLoading} placeholder="Paste candidate résumé text here…" style={textareaStyle(!!resumeText)} />
                  {resumeText && (
                    <div style={{ position: 'absolute', bottom: 10, right: 12, fontSize: 11, color: '#a8a29e', fontFamily: "'DM Mono',monospace" }}>
                      {resumeText.split(/\s+/).filter(Boolean).length}w
                    </div>
                  )}
                </div>
              )}

              {/* Submit */}
              <button type="submit" disabled={!canSubmit} style={{
                borderRadius: 10, padding: '12px 0', fontWeight: 600, fontSize: 14, border: 'none',
                cursor: canSubmit ? 'pointer' : 'not-allowed',
                background: canSubmit ? 'linear-gradient(135deg,#c9a96e,#a07840)' : '#e7e3de',
                color: canSubmit ? '#fff' : '#a8a29e',
                display: 'flex', alignItems: 'center', justifyContent: 'center', gap: 8,
                transition: 'all 0.2s', boxShadow: canSubmit ? '0 2px 12px rgba(160,120,64,0.3)' : 'none',
                letterSpacing: '-0.01em', fontFamily: "'DM Sans',sans-serif"
              }}>
                {isLoading
                  ? <><Loader2 style={{ width: 16, height: 16 }} className="spin" /> Analysing…</>
                  : <><Sparkles style={{ width: 16, height: 16 }} /> Run Semantic Analysis</>}
              </button>
            </form>
          </div>

          {/* ── Right Panel ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 16 }}>

            {rejectionError && (
              <div style={{ background: '#fff7ed', border: '1px solid #fed7aa', borderRadius: 12, padding: 16, display: 'flex', gap: 10, alignItems: 'flex-start' }}>
                <AlertCircle style={{ width: 16, height: 16, color: '#c2410c', flexShrink: 0, marginTop: 1 }} />
                <div>
                  <div style={{ fontSize: 13, fontWeight: 600, color: '#c2410c', marginBottom: 2 }}>Document rejected</div>
                  <div style={{ fontSize: 12, color: '#9a3412' }}>{rejectionError}</div>
                  <div style={{ fontSize: 12, color: '#a8a29e', marginTop: 4 }}>Your matcher requires a minimum word count. Paste more text and try again.</div>
                </div>
              </div>
            )}

            {!results && !isLoading && (
              <div style={{ minHeight: 400, border: '1px dashed rgba(0,0,0,0.1)', borderRadius: 16, display: 'flex', flexDirection: 'column', alignItems: 'center', justifyContent: 'center', gap: 12, color: '#a8a29e' }}>
                <div style={{ width: 56, height: 56, borderRadius: 16, background: '#f0ede8', display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
                  <Sparkles style={{ width: 24, height: 24, color: '#c9a96e', opacity: 0.7 }} />
                </div>
                <div style={{ textAlign: 'center' }}>
                  <div style={{ fontWeight: 500, fontSize: 14, color: '#78716c' }}>No results yet</div>
                  <div style={{ fontSize: 13, marginTop: 4 }}>Configure parameters and run the analysis</div>
                </div>
              </div>
            )}

            {isLoading && (
              <div style={{ minHeight: 300, border: '1px solid rgba(0,0,0,0.07)', borderRadius: 16, background: '#fff', display: 'flex', alignItems: 'center', justifyContent: 'center', boxShadow: '0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.06)' }}>
                <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 12 }}>
                  <Loader2 style={{ width: 28, height: 28, color: '#c9a96e' }} className="spin" />
                  <span style={{ fontSize: 14, color: '#78716c', fontWeight: 500 }}>Running semantic inference…</span>
                </div>
              </div>
            )}

            {failedFiles && (
              <div style={{ background: '#fff1f2', border: '1px solid #fecdd3', borderRadius: 12, padding: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 8, marginBottom: 8 }}>
                  <AlertCircle style={{ width: 15, height: 15, color: '#be123c' }} />
                  <span style={{ fontSize: 13, fontWeight: 600, color: '#be123c' }}>Processing failures</span>
                </div>
                {failedFiles.map((f, i) => (
                  <div key={i} style={{ fontSize: 12, color: '#9f1239', fontFamily: "'DM Mono',monospace", background: 'rgba(190,18,60,0.05)', borderRadius: 6, padding: '4px 8px', marginTop: 4 }}>
                    {f.filename}: {f.error}
                  </div>
                ))}
              </div>
            )}

            {results && (
              <>
                <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                  <span style={{ fontSize: 15, fontWeight: 700, color: '#1c1917', letterSpacing: '-0.02em' }}>Ranked Candidates</span>
                  <span style={{ fontSize: 12, color: '#78716c', background: '#fff', border: '1px solid rgba(0,0,0,0.08)', borderRadius: 20, padding: '3px 12px', fontFamily: "'DM Mono',monospace" }}>
                    {results.length} processed
                  </span>
                </div>

                {results.map((c, idx) => {
                  if (!c.match_data) return null;
                  const isExpanded = expandedCard === c.filename;
                  const score = c.match_data.overall_match_percentage ?? 0;
                  const scoreColor = score >= 80 ? '#059669' : score >= 50 ? '#d97706' : '#dc2626';
                  const badge = getRankBadge(idx);

                  return (
                    <div key={idx} className="result-card fade-in"
                      style={{ animationDelay: `${idx * 0.06}s`, background: '#fff', borderRadius: 14, border: '1px solid rgba(0,0,0,0.07)', overflow: 'hidden', transition: 'border-color 0.2s', boxShadow: '0 1px 3px rgba(0,0,0,0.06),0 4px 16px rgba(0,0,0,0.06)' }}>

                      <div onClick={() => toggleExpand(c.filename)}
                        style={{ padding: '16px 20px', display: 'flex', alignItems: 'center', justifyContent: 'space-between', cursor: 'pointer', userSelect: 'none', gap: 16 }}>
                        <div style={{ display: 'flex', alignItems: 'center', gap: 14, minWidth: 0 }}>
                          <div style={{ fontSize: 12, fontWeight: 700, color: '#d4c4a8', fontFamily: "'DM Mono',monospace", flexShrink: 0 }}>
                            #{String(idx + 1).padStart(2, '0')}
                          </div>
                          <ScoreRing score={score} />
                          <div style={{ minWidth: 0 }}>
                            <div style={{ display: 'flex', alignItems: 'center', gap: 8, flexWrap: 'wrap' }}>
                              <span style={{ fontWeight: 600, fontSize: 14, color: '#1c1917', overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap', maxWidth: 260 }}>
                                {c.filename}
                              </span>
                              {badge && (
                                <span style={{ fontSize: 11, fontWeight: 600, borderRadius: 20, padding: '2px 8px', border: `1px solid ${badge.border}`, background: badge.bg, color: badge.color, display: 'flex', alignItems: 'center', gap: 4, flexShrink: 0 }}>
                                  {badge.icon}{badge.label}
                                </span>
                              )}
                            </div>
                            <div style={{ marginTop: 6 }}>
                              <div className="progress-track" style={{ width: 200 }}>
                                <div style={{ height: '100%', borderRadius: 9999, background: scoreColor, width: `${score}%`, transition: 'width 0.8s cubic-bezier(0.4,0,0.2,1)' }} />
                              </div>
                            </div>
                            {c.match_data.density_penalty_applied && (
                              <span style={{ fontSize: 11, color: '#dc2626', display: 'flex', alignItems: 'center', gap: 4, marginTop: 4 }}>
                                <AlertCircle style={{ width: 11, height: 11 }} /> Keyword stuffing penalty applied
                              </span>
                            )}
                          </div>
                        </div>
                        <div style={{ color: '#c4b9a8', flexShrink: 0 }}>
                          {isExpanded ? <ChevronUp style={{ width: 18, height: 18 }} /> : <ChevronDown style={{ width: 18, height: 18 }} />}
                        </div>
                      </div>

                      {isExpanded && (
                        <div style={{ borderTop: '1px solid rgba(0,0,0,0.06)', padding: 20, background: '#fafaf9', display: 'flex', flexDirection: 'column', gap: 20 }}>

                          <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 10 }}>
                            {[
                              { label: 'Semantic Confidence', val: c.match_data.semantic_confidence ?? 0, accent: '#c9a96e' },
                              { label: 'Coherence Gap', val: c.match_data.coherence_gap ?? 0, accent: '#94a3b8' },
                            ].map(({ label, val, accent }) => (
                              <div key={label} style={{ background: 'rgba(255,255,255,0.6)', border: '1px solid rgba(0,0,0,0.06)', borderRadius: 10, padding: '12px 16px' }}>
                                <div style={{ fontSize: 11, fontWeight: 500, color: '#a8a29e', letterSpacing: '0.04em', textTransform: 'uppercase', marginBottom: 4 }}>{label}</div>
                                <div style={{ fontSize: 20, fontWeight: 700, color: '#1c1917', fontFamily: "'DM Mono',monospace" }}>{val.toFixed(3)}</div>
                                <div className="progress-track" style={{ marginTop: 8 }}>
                                  <div style={{ height: '100%', borderRadius: 9999, background: accent, width: `${Math.min(val * 100, 100)}%`, transition: 'width 0.8s cubic-bezier(0.4,0,0.2,1)' }} />
                                </div>
                              </div>
                            ))}
                          </div>

                          <div style={{ display: 'flex', flexDirection: 'column', gap: 14 }}>
                            {[
                              { icon: <CheckCircle2 style={{ width: 14, height: 14 }} />, label: 'Matched Skills', color: '#059669', items: c.match_data.skill_gap_report?.matched ?? [], cls: 'tag-matched', empty: 'No overlapping skills detected' },
                              { icon: <XCircle style={{ width: 14, height: 14 }} />, label: 'Missing Skills', color: '#be123c', items: c.match_data.skill_gap_report?.missing ?? [], cls: 'tag-missing', empty: 'All required skills satisfied' },
                              ...(c.match_data.skill_gap_report?.bonus?.length > 0
                                ? [{ icon: <Star style={{ width: 14, height: 14 }} />, label: 'Bonus Skills', color: '#2563eb', items: c.match_data.skill_gap_report.bonus, cls: 'tag-bonus', empty: '' }]
                                : [])
                            ].map(({ icon, label, color, items, cls, empty }) => (
                              <div key={label}>
                                <div style={{ display: 'flex', alignItems: 'center', gap: 6, marginBottom: 8, color }}>
                                  {icon}
                                  <span style={{ fontSize: 12, fontWeight: 600 }}>{label} ({items.length})</span>
                                </div>
                                <div style={{ display: 'flex', flexWrap: 'wrap', gap: 6 }}>
                                  {items.length > 0
                                    ? items.map(s => <span key={s} className={cls} style={{ padding: '3px 10px', borderRadius: 20, fontSize: 12, fontWeight: 500, fontFamily: "'DM Mono',monospace" }}>{s}</span>)
                                    : <span style={{ fontSize: 12, color: '#a8a29e', fontStyle: 'italic' }}>{empty}</span>}
                                </div>
                              </div>
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </>
            )}
          </div>
        </div>
      </main>
    </div>
  );
}