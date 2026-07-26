const $ = (id) => document.getElementById(id);
let currentReport = null;

function setMessage(text, kind = '') {
  const el = $('message'); el.textContent = text; el.className = `message ${kind}`;
}
function busy(state, text = 'Working…') {
  ['analyseBtn','humanizeBtn','fileInput'].forEach(id => $(id).disabled = state);
  if (state) setMessage(text);
}
async function api(url, options = {}) {
  const response = await fetch(url, options);
  if (!response.ok) {
    let detail = `Request failed (${response.status})`;
    try { const data = await response.json(); detail = data.detail || detail; } catch (_) {}
    throw new Error(detail);
  }
  return response;
}
function updateDashboard(report) {
  currentReport = report;
  const natural = report.naturalness_percentage ?? 0;
  $('naturalScore').textContent = `${natural}%`;
  $('concernScore').textContent = `${report.style_concern_percentage ?? 0}%`;
  $('wordCount').textContent = report.metrics?.word_count ?? 0;
  $('sentenceCount').textContent = report.metrics?.sentence_count ?? 0;
  $('highRisk').textContent = report.high_risk_segments ?? 0;
  $('moderateRisk').textContent = report.moderate_risk_segments ?? 0;
  $('scoreRing').style.background = `conic-gradient(var(--brand) ${natural * 3.6}deg,#dfe8f3 0deg)`;
  $('disclaimer').textContent = report.disclaimer || '';
  $('highlightedText').classList.remove('empty-state');
  $('highlightedText').innerHTML = report.highlighted_html || '';
  bindSegments();
  renderFindings(report.segments || []);
  renderMetrics(report.metrics || {});
}
function bindSegments() {
  document.querySelectorAll('.risk-segment').forEach(el => {
    const show = () => {
      $('segmentTip').innerHTML = `<b>${el.dataset.risk}% style concern.</b> ${el.dataset.reasons}`;
    };
    el.addEventListener('click', show); el.addEventListener('focus', show);
  });
}
function renderFindings(segments) {
  const flagged = segments.filter(s => ['high','moderate','low'].includes(s.band));
  if (!flagged.length) { $('findingsList').className='findings-list empty-state'; $('findingsList').textContent='No sentence-level concerns were detected.'; return; }
  $('findingsList').className='findings-list';
  $('findingsList').innerHTML = flagged.map(s => `<article class="finding ${s.band}"><div class="finding-head"><span>Sentence ${s.index + 1}</span><span>${s.risk}% concern</span></div><p>${escapeHtml(s.text)}</p><ul>${s.reasons.map(r=>`<li>${escapeHtml(r)}</li>`).join('')}</ul></article>`).join('');
}
function labelize(key) { return key.replaceAll('_',' ').replace(/\b\w/g, c=>c.toUpperCase()); }
function renderMetrics(metrics) {
  const omit = new Set(['repeated_frames']);
  const entries = Object.entries(metrics).filter(([k,v]) => !omit.has(k) && (typeof v === 'number' || typeof v === 'boolean' || typeof v === 'string'));
  $('metricsTable').className='metrics-table';
  $('metricsTable').innerHTML = entries.map(([k,v]) => `<div class="metric-card"><b>${escapeHtml(String(v))}</b><span>${escapeHtml(labelize(k))}</span></div>`).join('');
}
function escapeHtml(value) { const d=document.createElement('div'); d.textContent=value; return d.innerHTML; }
async function analyse() {
  const text = $('sourceText').value.trim(); if (!text) return setMessage('Add text before analysing.', 'error');
  busy(true, 'Analysing scholarly voice…');
  try { const response = await api('/api/analyse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})}); updateDashboard(await response.json()); setMessage('Analysis completed.', 'success'); }
  catch(e){ setMessage(e.message,'error'); } finally { busy(false); }
}
async function humanize() {
  const text = $('sourceText').value.trim(); if (!text) return setMessage('Add text before humanising.', 'error');
  busy(true, 'Applying protected scholarly refinement…');
  try {
    const response = await api('/api/humanize',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,mode:$('mode').value,use_model:$('useModel').checked})});
    const data = await response.json(); $('revisedText').value=data.text; updateDashboard(data.report); activateTab('revised');
    const modelNote = data.model_refiner?.applied ? ' Model-assisted refinement passed preservation checks.' : '';
    setMessage(`Humanisation completed.${modelNote}`, 'success');
  } catch(e){ setMessage(e.message,'error'); } finally { busy(false); }
}
async function uploadFile(file) {
  if (!file) return; busy(true,'Reading document…');
  const form = new FormData(); form.append('file',file);
  try { const response=await api('/api/upload',{method:'POST',body:form}); const data=await response.json(); $('sourceText').value=data.text; $('revisedText').value=''; updateDashboard(data.report); setMessage(`${data.filename} loaded.`, 'success'); }
  catch(e){ setMessage(e.message,'error'); } finally { busy(false); $('fileInput').value=''; }
}
function activateTab(name) { document.querySelectorAll('.tab').forEach(t=>t.classList.toggle('active',t.dataset.tab===name)); document.querySelectorAll('.panel').forEach(p=>p.classList.remove('active')); $(`${name}Panel`).classList.add('active'); }
async function exportFile(url, annotated, filename) {
  const text = ($('revisedText').value || $('sourceText').value).trim(); if (!text) return setMessage('There is no text to export.','error');
  try { const response=await api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,title:'Scholarly Humanizer Output',annotated})}); const blob=await response.blob(); const link=document.createElement('a'); link.href=URL.createObjectURL(blob); link.download=filename; link.click(); URL.revokeObjectURL(link.href); }
  catch(e){ setMessage(e.message,'error'); }
}
$('analyseBtn').addEventListener('click',analyse); $('humanizeBtn').addEventListener('click',humanize); $('fileInput').addEventListener('change',e=>uploadFile(e.target.files[0]));
$('clearBtn').addEventListener('click',()=>{ $('sourceText').value=''; $('revisedText').value=''; currentReport=null; $('highlightedText').textContent='Analyse text to colour the portions that need attention.'; $('highlightedText').className='document-view empty-state'; renderFindings([]); renderMetrics({}); setMessage('Workspace cleared.'); });
$('copyBtn').addEventListener('click',async()=>{ const text=$('revisedText').value||$('sourceText').value; await navigator.clipboard.writeText(text); setMessage('Text copied.','success'); });
$('docxBtn').addEventListener('click',()=>exportFile('/api/export/docx',false,'scholarly_humanized_text.docx'));
$('annotatedDocxBtn').addEventListener('click',()=>exportFile('/api/export/docx',true,'scholarly_voice_diagnostic.docx'));
$('htmlBtn').addEventListener('click',()=>exportFile('/api/export/html',true,'scholarly_voice_diagnostic.html'));
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>activateTab(tab.dataset.tab)));
fetch('/api/status').then(r=>r.json()).then(data=>{ $('modelStatus').textContent=data.model.message; $('useModel').disabled=!data.model.configured; }).catch(()=>{$('modelStatus').textContent='Local protected refinement active'; $('useModel').disabled=true;});
