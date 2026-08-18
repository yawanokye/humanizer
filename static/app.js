const $ = (id) => document.getElementById(id);
let currentReport = null;

function setMessage(text, kind = '') {
  const el = $('message');
  if (!el) return;
  el.textContent = text;
  el.className = `message ${kind}`;
}

function busy(state, text = 'Working…') {
  ['analyseBtn','humanizeBtn','fileInput'].forEach(id => {
    const el = $(id);
    if (el) el.disabled = state;
  });
  if (state) setMessage(text);
}

async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try {
      const data = await response.json();
      detail = data.detail || detail;
    } catch (_) {}
    throw new Error(detail);
  }
  return response;
}

function escapeHtml(value) {
  const d = document.createElement('div');
  d.textContent = value ?? '';
  return d.innerHTML;
}

function updateDashboard(report, comparison = null) {
  currentReport = report;
  const ai = Number(report.ai_detection_percentage ?? 0);
  const humanLike = Number(report.human_like_style_percentage ?? (100 - ai));
  const detector = report.ai_detector || {};

  if ($('aiScore')) $('aiScore').textContent = `${ai}%`;
  if ($('humanLikeScore')) $('humanLikeScore').textContent = `${humanLike}%`;
  if (!comparison && $('humanLikeGain')) $('humanLikeGain').textContent = '';
  if (!comparison && $('aiGain')) $('aiGain').textContent = '';
  if ($('aiVerdict')) $('aiVerdict').textContent = detector.verdict || report.ai_verdict || '—';
  if ($('aiConfidence')) $('aiConfidence').textContent = detector.confidence || report.ai_confidence || '—';
  if ($('forensicScore')) $('forensicScore').textContent = `${Number(detector.overall_score ?? report.ai_score ?? 0)} / ${Number(detector.max_score ?? report.ai_score_max ?? 27)}`;
  if ($('wordCount')) $('wordCount').textContent = report.metrics?.word_count ?? 0;
  if ($('sentenceCount')) $('sentenceCount').textContent = report.metrics?.sentence_count ?? 0;
  if ($('activeSignals')) $('activeSignals').textContent = `${report.active_signal_categories ?? 0}/9`;
  if ($('evidenceCount')) $('evidenceCount').textContent = report.signal_evidence_items ?? 0;
  if ($('aiScoreBar')) $('aiScoreBar').style.width = `${Math.max(0, Math.min(100, ai))}%`;
  if ($('humanLikeScoreBar')) $('humanLikeScoreBar').style.width = `${Math.max(0, Math.min(100, humanLike))}%`;
  if (comparison?.ai && $('aiGain')) {
    const reduction = Number(comparison.ai.reduction ?? 0);
    $('aiGain').textContent = reduction > 0 ? `▼ ${reduction} points after rewrite` : reduction < 0 ? `▲ ${Math.abs(reduction)} points after rewrite` : 'no change after rewrite';
    $('aiGain').className = `score-change ai-gain${reduction < 0 ? ' negative' : ''}`;
  }
  if (comparison?.humanLike && $('humanLikeGain')) {
    const gain = Number(comparison.humanLike.gain ?? 0);
    $('humanLikeGain').textContent = gain > 0 ? `▲ +${gain} points after rewrite` : gain < 0 ? `▼ ${Math.abs(gain)} points after rewrite` : 'no change after rewrite';
    $('humanLikeGain').className = `score-change humanlike-gain${gain < 0 ? ' negative' : ''}`;
  }
  if ($('detectorVariability')) { const span = $('detectorVariability').querySelector('span'); if (span) span.textContent = report.detector_variability_notice || detector.detector_variability_notice || 'AI-writing detectors can disagree substantially, especially on formal academic prose. Use the score as a style-screening indicator, not proof of authorship.'; }
  if ($('disclaimer')) $('disclaimer').textContent = report.disclaimer || '';

  if ($('highlightedText')) {
    $('highlightedText').classList.remove('empty-state');
    $('highlightedText').innerHTML = report.highlighted_html || '';
  }
  bindSegments();
  renderDetector(detector);
  renderFindings(report.segments || []);
  renderMetrics(report.metrics || {});
}

function bindSegments() {
  document.querySelectorAll('.risk-segment').forEach(el => {
    const show = () => {
      const tip = $('segmentTip');
      if (tip) tip.innerHTML = `<b>${escapeHtml(el.dataset.risk)}% AI-style signal.</b> ${escapeHtml(el.dataset.reasons || '')}`;
    };
    el.addEventListener('click', show);
    el.addEventListener('focus', show);
  });
}

function scoreTone(score) {
  if (score >= 3) return 'high';
  if (score >= 2) return 'moderate';
  if (score >= 1) return 'low';
  return 'natural';
}

function renderDetector(detector) {
  const target = $('detectorSignals');
  if (!target) return;
  const signals = detector?.signals || [];
  if (!signals.length) {
    target.className = 'detector-signals empty-state';
    target.textContent = 'Run AI detection to see the nine-signal breakdown.';
    return;
  }

  const header = `
    <section class="detector-summary">
      <div><small>AI signal index</small><strong>${Number(detector.ai_detection_percentage || 0)}%</strong></div>
      <div><small>Forensic category score</small><strong>${Number(detector.overall_score || 0)} / ${Number(detector.max_score || 27)}</strong></div>
      <div><small>Signal level</small><strong>${escapeHtml(detector.signal_level || detector.verdict || '—')}</strong></div>
      <div><small>Confidence</small><strong>${escapeHtml(detector.confidence || '—')}</strong></div>
    </section>`;

  const cards = signals.map(signal => {
    const score = Number(signal.score || 0);
    const pct = Math.round((score / 3) * 100);
    const evidence = (signal.evidence || []).length
      ? `<ul>${signal.evidence.map(item => `<li><span class="severity ${escapeHtml(item.severity || 'weak')}">${escapeHtml(item.severity || 'weak')}</span>${escapeHtml(item.description || '')}</li>`).join('')}</ul>`
      : '<p class="no-signal">No signal crossed the threshold.</p>';
    return `<article class="signal-card ${scoreTone(score)}">
      <header><div><span class="signal-key">${escapeHtml(signal.key || '')}</span><h3>${escapeHtml(signal.name || '')}</h3></div><strong>${score}/3 <small>${pct}%</small></strong></header>
      <p>${escapeHtml(signal.summary || '')}</p>
      <div class="bar"><i style="width:${pct}%"></i></div>
      ${evidence}
    </article>`;
  }).join('');

  const notes = (detector.calibration_notes || []).map(note => `<li>${escapeHtml(note)}</li>`).join('');
  const narrative = `<section class="detector-explanation"><h3>What gave it away</h3><p>${escapeHtml(detector.what_gave_it_away || '')}</p><h3>Calibration</h3><ul>${notes}</ul></section>`;

  target.className = 'detector-signals';
  target.innerHTML = header + `<div class="signal-grid">${cards}</div>` + narrative;
}

function renderFindings(segments) {
  const target = $('findingsList');
  if (!target) return;
  const flagged = segments.filter(s => ['high','moderate','low'].includes(s.band));
  if (!flagged.length) {
    target.className = 'findings-list empty-state';
    target.textContent = 'No sentence-level AI-style signals crossed the display threshold.';
    return;
  }
  target.className = 'findings-list';
  target.innerHTML = flagged.map(s => `<article class="finding ${s.band}"><div class="finding-head"><span>Sentence ${s.index + 1}</span><span>${s.risk}% AI signal</span></div><p>${escapeHtml(s.text)}</p><ul>${(s.reasons || []).map(r=>`<li>${escapeHtml(r)}</li>`).join('')}</ul></article>`).join('');
}

function labelize(key) {
  return key.replaceAll('_',' ').replace(/\b\w/g, c=>c.toUpperCase());
}

function renderMetrics(metrics) {
  const target = $('metricsTable');
  if (!target) return;
  const omit = new Set(['repeated_frames', 'naturalness_score']);
  const entries = Object.entries(metrics).filter(([k,v]) => !omit.has(k) && (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string'));
  target.className='metrics-table';
  target.innerHTML = entries.map(([k,v]) => `<div class="metric-card"><b>${escapeHtml(String(v))}</b><span>${escapeHtml(labelize(k))}</span></div>`).join('');
}

async function analyse() {
  const text = $('sourceText')?.value.trim();
  if (!text) return setMessage('Add text before running AI detection.', 'error');
  busy(true, 'Running nine-signal AI detection…');
  try {
    const response = await api('/api/analyse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    updateDashboard(await response.json());
    activateTab('detector');
    setMessage('AI detection completed. Review the signal breakdown and sentence evidence together.', 'success');
  } catch(e) {
    setMessage(e.message,'error');
  } finally {
    busy(false);
  }
}

async function humanize() {
  const text = $('sourceText')?.value.trim();
  if (!text) return setMessage('Add text before humanising.', 'error');
  busy(true, 'Applying protected scholarly refinement…');
  try {
    const response = await api('/api/humanize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,mode:$('mode')?.value || 'balanced',engine:$('engine')?.value || 'engine1',engine2_model:$('engine2Model')?.value || 'gpt-5.6-terra'})});
    const data = await response.json();
    if ($('revisedText')) $('revisedText').value=data.text;
    const aiImprovement = data.ai_signal_improvement || {};
    const humanLikeImprovement = data.human_like_style_improvement || {
      before: 100 - Number(aiImprovement.before ?? data.original_report?.ai_detection_percentage ?? 0),
      after: 100 - Number(aiImprovement.after ?? data.report?.ai_detection_percentage ?? 0),
      gain: Number(aiImprovement.reduction ?? 0),
    };
    updateDashboard(data.report, {humanLike: humanLikeImprovement, ai: aiImprovement});
    activateTab('revised');
    const engineNote = data.actual_engine === 'engine1_fallback'
      ? ` Engine 2 was unavailable, so Engine 1 fallback was used. ${data.engine_2?.reason || ''}`
      : data.selected_engine === 'engine2'
        ? (data.engine_2?.applied ? ' Engine 2 API rewrite passed preservation and rewrite-quality checks.' : ` ${data.engine_2?.reason || 'Engine 2 did not apply changes.'}`)
        : ' Engine 1 local rewrite completed without an API call.';
    const humanLikeNote = ` Human-like style ${Number(humanLikeImprovement.before ?? 0)}% → ${Number(humanLikeImprovement.after ?? 0)}%.`;
    const aiNote = ` AI signal index ${Number(aiImprovement.before ?? data.original_report?.ai_detection_percentage ?? 0)}% → ${Number(aiImprovement.after ?? data.report?.ai_detection_percentage ?? 0)}%.`;
    const outcome = data.changed ? 'Humanisation completed.' : 'No safe rewrite changes were made.';
    setMessage(`${outcome}${engineNote}${aiNote}${humanLikeNote}`, data.changed ? 'success' : '');
  } catch(e) {
    setMessage(e.message,'error');
  } finally {
    busy(false);
  }
}

async function uploadFile(file) {
  if (!file) return;
  busy(true,'Reading document and running AI detection…');
  const form = new FormData();
  form.append('file',file);
  try {
    const response=await api('/api/upload',{method:'POST',body:form});
    const data=await response.json();
    if ($('sourceText')) $('sourceText').value=data.text;
    if ($('revisedText')) $('revisedText').value='';
    updateDashboard(data.report);
    activateTab('detector');
    setMessage(`${data.filename} loaded and screened.`, 'success');
  } catch(e) {
    setMessage(e.message,'error');
  } finally {
    busy(false);
    if ($('fileInput')) $('fileInput').value='';
  }
}

function activateTab(name) {
  document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===name));
  document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active'));
  const panel = $(`${name}Panel`);
  if (panel) panel.classList.add('active');
}

async function exportFile(url, annotated, filename) {
  const text = (($('revisedText')?.value || $('sourceText')?.value) || '').trim();
  if (!text) return setMessage('There is no text to export.','error');
  try {
    const response=await api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,title:'Scholarly Humanizer Output',annotated})});
    const blob=await response.blob();
    const link=document.createElement('a');
    link.href=URL.createObjectURL(blob);
    link.download=filename;
    link.click();
    URL.revokeObjectURL(link.href);
  } catch(e) {
    setMessage(e.message,'error');
  }
}

const legacyUseModel = $('useModel');
const engineSelect = $('engine');
const engine2ModelSelect = $('engine2Model');
let engine2Configured = false;

function syncEngineControls() {
  const usingEngine2 = engineSelect?.value === 'engine2';
  if (legacyUseModel && engineSelect) legacyUseModel.checked = usingEngine2;
  const wrap = $('engine2ModelWrap');
  if (wrap) wrap.hidden = !usingEngine2;
  if (engine2ModelSelect) engine2ModelSelect.disabled = false;
  const hint = $('engine2Hint');
  if (hint && usingEngine2) {
    hint.textContent = engine2Configured
      ? 'Engine 2 is configured. Choose Terra for stronger quality or Luna for lower-cost high-volume rewriting.'
      : 'Engine 2 can be selected, but OPENAI_API_KEY must be added in Render before API rewriting can run.';
  }
  try {
    if (engineSelect) localStorage.setItem('humanizer_engine', engineSelect.value);
    if (engine2ModelSelect) localStorage.setItem('humanizer_engine2_model', engine2ModelSelect.value);
  } catch (_) {}
}

try {
  const savedEngine = localStorage.getItem('humanizer_engine');
  const savedModel = localStorage.getItem('humanizer_engine2_model');
  if (engineSelect && ['engine1','engine2'].includes(savedEngine)) engineSelect.value = savedEngine;
  if (engine2ModelSelect && ['gpt-5.6-terra','gpt-5.6-luna'].includes(savedModel)) engine2ModelSelect.value = savedModel;
} catch (_) {}

if (engineSelect) engineSelect.addEventListener('change', syncEngineControls);
if (engine2ModelSelect) engine2ModelSelect.addEventListener('change', syncEngineControls);
syncEngineControls();

$('analyseBtn')?.addEventListener('click',analyse);
$('humanizeBtn')?.addEventListener('click',humanize);
$('fileInput')?.addEventListener('change',e=>uploadFile(e.target.files[0]));
$('clearBtn')?.addEventListener('click',()=>{
  if ($('sourceText')) $('sourceText').value='';
  if ($('revisedText')) $('revisedText').value='';
  currentReport=null;
  if ($('aiScore')) $('aiScore').textContent='—';
  if ($('humanLikeScore')) $('humanLikeScore').textContent='—';
  if ($('humanLikeGain')) $('humanLikeGain').textContent='';
  if ($('aiGain')) $('aiGain').textContent='';
  if ($('aiScoreBar')) $('aiScoreBar').style.width='0%';
  if ($('humanLikeScoreBar')) $('humanLikeScoreBar').style.width='0%';
  if ($('aiVerdict')) $('aiVerdict').textContent='—';
  if ($('aiConfidence')) $('aiConfidence').textContent='—';
  if ($('forensicScore')) $('forensicScore').textContent='—';
  if ($('wordCount')) $('wordCount').textContent='0';
  if ($('sentenceCount')) $('sentenceCount').textContent='0';
  if ($('activeSignals')) $('activeSignals').textContent='0/9';
  if ($('evidenceCount')) $('evidenceCount').textContent='0';
  if ($('highlightedText')) {
    $('highlightedText').textContent='Run AI detection to colour sentences by AI-style signal strength.';
    $('highlightedText').className='document-view empty-state';
  }
  renderDetector({});
  renderFindings([]);
  renderMetrics({});
  activateTab('detector');
  setMessage('Workspace cleared.');
});
$('copyBtn')?.addEventListener('click',async()=>{
  const text=$('revisedText')?.value||$('sourceText')?.value||'';
  await navigator.clipboard.writeText(text);
  setMessage('Text copied.','success');
});
$('docxBtn')?.addEventListener('click',()=>exportFile('/api/export/docx',false,'scholarly_humanized_text.docx'));
$('annotatedDocxBtn')?.addEventListener('click',()=>exportFile('/api/export/docx',true,'ai_signal_diagnostic.docx'));
$('htmlBtn')?.addEventListener('click',()=>exportFile('/api/export/html',true,'ai_signal_diagnostic.html'));
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>activateTab(tab.dataset.tab)));

fetch('/api/status').then(r=>r.json()).then(data=>{
  if ($('modelStatus')) $('modelStatus').textContent=data.model.message;
  const engine2 = data.engines?.engine2;
  engine2Configured = Boolean(engine2?.configured);
  const opt = $('engine2Option');
  if (opt) {
    // Never disable Engine 2. Configuration status should explain availability, not trap the selector.
    opt.disabled = false;
    opt.textContent = engine2Configured ? 'Engine 2, API rewrite' : 'Engine 2, API rewrite, API key required';
  }
  syncEngineControls();
}).catch(()=>{
  engine2Configured = false;
  const opt = $('engine2Option');
  if (opt) opt.disabled = false;
  if ($('modelStatus')) $('modelStatus').textContent='Engine 1 active. Engine 2 is selectable but its API configuration could not be verified.';
  syncEngineControls();
});
