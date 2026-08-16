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

function updateDashboard(report) {
  currentReport = report;
  const ai = Number(report.ai_detection_percentage ?? 0);
  const natural = Number(report.naturalness_percentage ?? 0);
  const detector = report.ai_detector || {};

  if ($('aiScore')) $('aiScore').textContent = `${ai}%`;
  if ($('naturalScore')) $('naturalScore').textContent = `${natural}%`;
  if ($('naturalGain') && !report.naturalness_improvement) $('naturalGain').textContent = '';
  if ($('aiVerdict')) $('aiVerdict').textContent = detector.verdict || report.ai_verdict || '—';
  if ($('aiConfidence')) $('aiConfidence').textContent = detector.confidence || report.ai_confidence || '—';
  if ($('aiFraction')) $('aiFraction').textContent = detector.ai_edited_fraction || report.ai_edited_fraction || '—';
  if ($('wordCount')) $('wordCount').textContent = report.metrics?.word_count ?? 0;
  if ($('sentenceCount')) $('sentenceCount').textContent = report.metrics?.sentence_count ?? 0;
  if ($('highRisk')) $('highRisk').textContent = report.high_risk_segments ?? 0;
  if ($('moderateRisk')) $('moderateRisk').textContent = report.moderate_risk_segments ?? 0;
  if ($('scoreRing')) $('scoreRing').style.background = `conic-gradient(var(--detector) ${ai * 3.6}deg,#dfe8f3 0deg)`;
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
      <div><small>Overall score</small><strong>${Number(detector.overall_score || 0)} / ${Number(detector.max_score || 27)}</strong></div>
      <div><small>Verdict</small><strong>${escapeHtml(detector.verdict || '—')}</strong></div>
      <div><small>Confidence</small><strong>${escapeHtml(detector.confidence || '—')}</strong></div>
      <div><small>AI-edited estimate</small><strong>${escapeHtml(detector.ai_edited_fraction || '—')}</strong></div>
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
  const omit = new Set(['repeated_frames']);
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
    const response = await api('/api/humanize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,mode:$('mode')?.value || 'balanced',engine:$('engine')?.value || 'engine1'})});
    const data = await response.json();
    if ($('revisedText')) $('revisedText').value=data.text;
    updateDashboard(data.report);
    activateTab('revised');
    const improvement = data.naturalness_improvement || {};
    const gain = Number(improvement.gain || 0);
    if ($('naturalGain')) {
      $('naturalGain').textContent = gain > 0 ? `▲ +${gain} after rewrite` : 'no reduction';
      $('naturalGain').className = `natural-gain${gain < 0 ? ' negative' : ''}`;
    }
    const engineNote = data.selected_engine === 'engine2'
      ? (data.engine_2?.applied ? ' Engine 2 API rewrite passed preservation and naturalness checks.' : ` ${data.engine_2?.reason || 'Engine 2 did not apply changes.'}`)
      : ' Engine 1 local rewrite completed without an API call.';
    const scoreNote = ` Naturalness ${Number(improvement.before ?? data.original_report?.naturalness_percentage ?? 0)}% → ${Number(improvement.after ?? data.report?.naturalness_percentage ?? 0)}%.`;
    setMessage(`Humanisation completed.${engineNote}${scoreNote} The revised text has also been rescored by the AI detector.`, 'success');
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
function syncLegacyEngineFlag() {
  if (legacyUseModel && engineSelect) legacyUseModel.checked = engineSelect.value === 'engine2';
}
if (engineSelect) engineSelect.addEventListener('change', syncLegacyEngineFlag);
syncLegacyEngineFlag();

$('analyseBtn')?.addEventListener('click',analyse);
$('humanizeBtn')?.addEventListener('click',humanize);
$('fileInput')?.addEventListener('change',e=>uploadFile(e.target.files[0]));
$('clearBtn')?.addEventListener('click',()=>{
  if ($('sourceText')) $('sourceText').value='';
  if ($('revisedText')) $('revisedText').value='';
  currentReport=null;
  if ($('aiScore')) $('aiScore').textContent='—';
  if ($('naturalScore')) $('naturalScore').textContent='—';
  if ($('naturalGain')) $('naturalGain').textContent='';
  if ($('aiVerdict')) $('aiVerdict').textContent='—';
  if ($('aiConfidence')) $('aiConfidence').textContent='—';
  if ($('aiFraction')) $('aiFraction').textContent='—';
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
  const opt = $('engine2Option');
  if (opt) {
    opt.disabled = !engine2?.configured;
    opt.textContent = engine2?.configured ? 'Engine 2, API rewrite' : 'Engine 2, API rewrite (not configured)';
  }
}).catch(()=>{
  if ($('modelStatus')) $('modelStatus').textContent='Engine 1 local rewrite active. Engine 2 not configured.';
});
