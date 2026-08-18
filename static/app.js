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
  if ($('forensicScore')) $('forensicScore').textContent = `${Number(detector.overall_score ?? report.ai_score ?? 0).toFixed(1)} / ${Number(detector.max_score ?? report.ai_score_max ?? 27)}`;
  if ($('humannessCounter')) $('humannessCounter').textContent = `${Number(detector.humanness_counter_score ?? 0)} point${Number(detector.humanness_counter_score ?? 0) === 1 ? '' : 's'} · confidence only`;
  if ($('decisionStatus')) $('decisionStatus').textContent = report.decision_status || detector.decision_status || '—';
  if ($('scoreSource')) $('scoreSource').textContent = (report.score_source || detector.score_source || '—').replaceAll('_',' ');
  if ($('wordCount')) $('wordCount').textContent = report.metrics?.word_count ?? 0;
  if ($('sentenceCount')) $('sentenceCount').textContent = report.metrics?.sentence_count ?? 0;
  if ($('activeSignals')) $('activeSignals').textContent = `${report.active_signal_categories ?? 0}/9`;
  if ($('evidenceCount')) $('evidenceCount').textContent = report.signal_evidence_items ?? 0;
  if ($('proseSegments')) $('proseSegments').textContent = report.prose_segment_count ?? detector.segment_count ?? 0;
  if ($('flaggedSegments')) $('flaggedSegments').textContent = report.flagged_prose_segments ?? detector.flagged_segment_count ?? 0;
  if ($('statisticalFingerprint')) $('statisticalFingerprint').textContent = `${Number(report.statistical_fingerprint_percentage ?? detector.statistical_fingerprint_percentage ?? 0)}%`;
  if ($('engineSignalTarget')) { const active=(detector.signals||[]).filter(x=>Number(x.score||0)>0).map(x=>x.key); $('engineSignalTarget').textContent=active.length ? active.join(', ') : 'None'; }
  if ($('calibrationState')) $('calibrationState').textContent = report.calibration?.trained ? `Trained · ${Number(report.calibration.sample_count||0)} samples` : 'Fallback';
  if ($('referenceLmState')) $('referenceLmState').textContent = report.reference_lm?.scored ? 'True LM' : (report.reference_lm?.enabled ? 'Unavailable' : 'Proxy');
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
  if ($('detectorVariability')) { const span = $('detectorVariability').querySelector('span'); if (span) span.textContent = report.detector_variability_notice || detector.detector_variability_notice || 'AI-writing detectors can disagree substantially, especially on formal academic prose. This app combines global and paragraph-level evidence. Use the score as a style-screening indicator, not proof of authorship.'; }
  if ($('disclaimer')) $('disclaimer').textContent = report.disclaimer || '';

  if ($('highlightedText')) {
    $('highlightedText').classList.remove('empty-state');
    $('highlightedText').innerHTML = report.highlighted_html || '';
  }
  if ($('signalColouredText')) {
    $('signalColouredText').classList.remove('empty-state');
    $('signalColouredText').innerHTML = report.signal_coloured_html || '';
  }
  bindSegments();
  bindSignalColours();
  renderDetector(detector);
  renderStatistical(report.style_concern_categories || [], detector);
  renderCalibration(report);
  renderSectionProfile(report.section_profile || detector.section_profile || {});
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

function bindSignalColours() {
  document.querySelectorAll('.signal-text[data-signals]').forEach(el => {
    const show = () => {
      const tip = $('signalColourTip');
      if (!tip) return;
      const keys = (el.dataset.signals || '').split(',').filter(Boolean);
      tip.innerHTML = `<b>Signals ${escapeHtml(keys.join(', ') || '—')}.</b> ${escapeHtml(el.dataset.reasons || '')}`;
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
    target.textContent = 'Run AI detection to see the A–I forensic breakdown.';
    return;
  }

  const header = `
    <section class="detector-summary">
      <div><small>AI signal index</small><strong>${Number(detector.ai_detection_percentage || 0)}%</strong></div>
      <div><small>Weighted A–I score</small><strong>${Number(detector.overall_score || 0).toFixed(1)} / ${Number(detector.max_score || 27)}</strong></div>
      <div><small>Signal level</small><strong>${escapeHtml(detector.signal_level || detector.verdict || '—')}</strong></div>
      <div><small>Confidence</small><strong>${escapeHtml(detector.confidence || '—')}</strong></div>
    </section>`;

  const weights = detector.composite_weights || {};
  const calibrated = detector.score_source === 'calibrated_meta_classifier';
  const arithmetic = `<section class="score-arithmetic">
    <b>${calibrated ? 'Calibrated meta-classifier' : 'Transparent four-layer fallback'}</b>
    <span>Forensic A–I ${Number(detector.category_signal_percentage ?? 0)}% × ${Math.round(Number(weights.forensic ?? 0.25) * 100)}%</span>
    <span>Statistical fingerprint ${Number(detector.statistical_fingerprint_percentage ?? 0)}% × ${Math.round(Number(weights.statistical ?? 0.35) * 100)}%</span>
    <span>Paragraph profile ${Number(detector.segment_signal_percentage ?? 0)}% × ${Math.round(Number(weights.segments ?? 0.30) * 100)}%</span>
    <span>Document regularity ${Number(detector.consistency_signal_percentage ?? 0)}% × ${Math.round(Number(weights.document_consistency ?? 0.10) * 100)}%</span>
    ${calibrated ? `<span>Transparent fallback: ${Number(detector.fallback_ensemble_percentage ?? 0)}%</span><span>Learned probability: ${Number(detector.ai_detection_percentage ?? 0)}%</span>` : `<span>Corroboration bonus +${Number(detector.corroboration_bonus ?? 0)}</span><span>Composite = ${Number(detector.ai_detection_percentage ?? 0)}%</span>`}
    <span>Decision: ${escapeHtml(detector.decision_status || '—')} · ${escapeHtml(detector.decision_reason || '')}</span>
    <span>Human-context evidence ${Number(detector.humanness_counter_score ?? 0)} point(s), confidence only</span>
  </section>`;

  const cards = signals.map(signal => {
    const score = Number(signal.score || 0);
    const pct = Math.round((score / 3) * 100);
    const evidence = (signal.evidence || []).length
      ? `<ul>${signal.evidence.map(item => `<li><span class="severity ${escapeHtml(item.severity || 'weak')}">${escapeHtml(item.severity || 'weak')}</span>${escapeHtml(item.description || '')}</li>`).join('')}</ul>`
      : '<p class="no-signal">No signal crossed the threshold.</p>';
    return `<article class="signal-card ${scoreTone(score)}">
      <header><div><span class="signal-key">${escapeHtml(signal.key || '')}</span><h3>${escapeHtml(signal.name || '')}</h3></div><strong>${score}/3 <small>${pct}% · ×${Number(signal.weight ?? 1).toFixed(2)}</small></strong></header>
      <p>${escapeHtml(signal.summary || '')}</p>
      <div class="bar"><i style="width:${pct}%"></i></div>
      ${evidence}
    </article>`;
  }).join('');

  const segmentProfile = detector.segment_profile || [];
  const hotspots = [...segmentProfile].sort((a,b)=>Number(b.ai_signal||0)-Number(a.ai_signal||0)).filter(x=>Number(x.ai_signal||0)>0).slice(0,8);
  const segmentSummary = `<section class="segment-profile-summary">
    <header><div><h3>Paragraph-level profile</h3><p>${Number(detector.segment_count ?? 0)} prose segments screened. ${Number(detector.flagged_segment_count ?? 0)} reached 20% or more local signal.</p></div><strong>P90 ${Number(detector.segment_p90 ?? 0)}%</strong></header>
    ${hotspots.length ? `<div class="segment-hotspots">${hotspots.map(item=>`<article><b>Segment ${Number(item.segment||0)} · ${Number(item.ai_signal||0)}%</b><span>${escapeHtml(item.excerpt||'')}</span></article>`).join('')}</div>` : '<p class="no-signal">No paragraph-level hotspot crossed the local display threshold.</p>'}
  </section>`;
  const notes = (detector.calibration_notes || []).map(note => `<li>${escapeHtml(note)}</li>`).join('');
  const narrative = `<section class="detector-explanation"><h3>What gave it away</h3><p>${escapeHtml(detector.what_gave_it_away || '')}</p><h3>Calibration</h3><ul>${notes}</ul></section>`;

  target.className = 'detector-signals';
  target.innerHTML = header + arithmetic + segmentSummary + `<div class="signal-grid">${cards}</div>` + narrative;
}

function renderStatistical(groups, detector = {}) {
  const target = $('statisticalSignals');
  if (!target) return;
  if (!groups?.length) {
    target.className = 'concern-categories empty-state';
    target.textContent = 'Run AI detection to inspect the continuous statistical fingerprint.';
    return;
  }
  target.className = 'concern-categories';
  const ref = detector.reference_lm || {};
  const refBlock = `<section class="concern-group reference-lm-card">
    <header><div><h3>Reference-language-model probability diagnostics</h3><p>${escapeHtml(ref.message || 'Optional true token-probability layer is not configured.')}</p></div><strong>${ref.scored ? 'Active' : 'Optional'}</strong></header>
    ${ref.scored ? `<div class="reference-metrics"><span><b>${Number(ref.perplexity||0).toFixed(2)}</b>Perplexity</span><span><b>${Number(ref.surprisal_mean||0).toFixed(2)}</b>Mean surprisal</span><span><b>${Number(ref.surprisal_std||0).toFixed(2)}</b>Surprisal SD</span><span><b>${Math.round(Number(ref.low_surprisal_share||0)*100)}%</b>Low-surprisal tokens</span><span><b>${Number(ref.longest_low_surprisal_run||0)}</b>Longest predictable run</span></div>` : `<p class="no-signal">Install requirements-reference-lm.txt and enable REFERENCE_LM_ENABLED=true to calculate true token probabilities. Proxy metrics above remain active otherwise.</p>`}
  </section>`;
  target.innerHTML = groups.map(group => `
    <section class="concern-group">
      <header><div><h3>${escapeHtml(group.group || '')}</h3><p>${escapeHtml(group.description || '')}</p></div><strong>${Number(group.percentage || 0)}%</strong></header>
      ${(group.metrics || []).map(metric => `<article class="concern-metric">
        <div class="concern-row"><span>${escapeHtml(metric.label || '')}</span><b>${Number(metric.percentage || 0)}%</b></div>
        <div class="bar"><i style="width:${Math.max(0,Math.min(100,Number(metric.percentage||0)))}%"></i></div>
        <ul>${(metric.evidence || []).map(item=>`<li>${escapeHtml(item)}</li>`).join('')}</ul>
      </article>`).join('')}
    </section>`).join('') + refBlock;
}

function renderCalibration(report = {}) {
  const target = $('calibrationDetails');
  if (!target) return;
  const calibration = report.calibration || report.ai_detector?.calibration || {};
  const prediction = report.ai_detector?.calibration_prediction || null;
  const features = report.calibration_features || report.ai_detector?.calibration_features || {};
  target.className = 'calibration-details';
  const metrics = calibration.metrics || {};
  const metricCards = calibration.trained ? `<div class="reference-metrics"><span><b>${Number(calibration.sample_count||0)}</b>Training samples</span><span><b>${metrics.false_positive_rate !== undefined ? Math.round(Number(metrics.false_positive_rate)*100)+'%' : '—'}</b>Validation FPR</span><span><b>${metrics.false_negative_rate !== undefined ? Math.round(Number(metrics.false_negative_rate)*100)+'%' : '—'}</b>Validation FNR</span><span><b>${metrics.f1 !== undefined ? Number(metrics.f1).toFixed(2) : '—'}</b>Validation F1</span><span><b>${metrics.roc_auc !== undefined ? Number(metrics.roc_auc).toFixed(2) : '—'}</b>Validation ROC-AUC</span></div>` : '';
  const featureRows = Object.entries(features).map(([key,value])=>`<div><span>${escapeHtml(labelize(key))}</span><b>${Number(value).toFixed(2)}</b></div>`).join('');
  target.innerHTML = `<section class="calibration-card ${calibration.trained ? 'trained' : 'fallback'}"><header><div><h3>${calibration.trained ? `Labelled meta-classifier active · ${escapeHtml(calibration.model_type || 'learned model')}` : 'Transparent fallback active'}</h3><p>${escapeHtml(calibration.message || '')}</p></div><strong>${escapeHtml(report.decision_status || report.ai_detector?.decision_status || '—')}</strong></header>${metricCards}<p>${escapeHtml(report.decision_reason || report.ai_detector?.decision_reason || '')}</p>${prediction ? `<p><b>Learned probability:</b> ${Number(prediction.percentage||0)}% · fallback ensemble ${Number(report.ai_detector?.fallback_ensemble_percentage||0)}%</p>` : '<p>No corpus-trained model is installed, so the app does not pretend the score is empirically calibrated.</p>'}<details><summary>Feature vector used by the calibrator</summary><div class="calibration-feature-grid">${featureRows}</div></details></section>`;
}

function renderSectionProfile(profile = {}) {
  const target = $('sectionProfile');
  if (!target) return;
  const labels = {abstract:'Abstract / executive summary', intro_lit:'Introduction / literature', methods:'Methods', results:'Results', discussion:'Discussion', conclusion:'Conclusion / limitations', other:'Other prose'};
  const rows = Object.entries(labels).map(([key,label]) => {
    const item = profile[key] || {};
    const mean = Number(item.mean_signal || 0);
    return `<article class="section-profile-card"><header><div><h3>${escapeHtml(label)}</h3><p>${Number(item.segment_count||0)} sentence segments · ${Number(item.flagged_count||0)} flagged · ${Number(item.elevated_count||0)} elevated</p></div><strong>${mean.toFixed(1)}%</strong></header><div class="bar"><i style="width:${Math.max(0,Math.min(100,mean))}%"></i></div></article>`;
  }).join('');
  target.className = 'section-profile';
  target.innerHTML = `<section class="section-profile-summary"><header><div><h3>Section-aware academic profile</h3><p>The calibrator receives section-level signal means so Methods and Results are not assumed to behave like Discussion or reflective prose.</p></div><strong>Spread ${Number(profile.score_spread||0).toFixed(1)}</strong></header></section><div class="section-profile-grid">${rows}</div>`;
}

function renderRewriteAudit(data = null) {
  const target = $('rewriteAudit');
  if (!target) return;
  if (!data) {
    target.className = 'rewrite-audit empty-state';
    target.textContent = 'Run a humanizer to see the before/after signal map. Engine 3 shows which diagnosed signals it deliberately targeted.';
    return;
  }
  const engine = data.selected_engine || 'engine1';
  const engine3 = data.engine_3 || {};
  const signalMap = engine3.signal_map || [];
  const statisticalMap = engine3.statistical_map || [];
  const ai = data.ai_signal_improvement || {};
  const human = data.human_like_style_improvement || {};
  const signalRows = signalMap.length ? signalMap.map(item => `<div class="audit-row ${item.targeted?'targeted':''}"><b>${escapeHtml(item.key)}</b><span>${Number(item.before||0)}/3 → ${Number(item.after||0)}/3</span><em>${item.targeted?'targeted':'audit only'}</em></div>`).join('') : '<p class="no-signal">This engine is detector-independent, so there is no detector-target list.</p>';
  const statRows = statisticalMap.filter(item => Number(item.before||0) || Number(item.after||0)).slice(0,12).map(item => `<div class="audit-row"><b>${escapeHtml(labelize(item.key))}</b><span>${Number(item.before||0)}% → ${Number(item.after||0)}%</span><em>${Number(item.change||0)>0?'higher':Number(item.change||0)<0?'lower':'same'}</em></div>`).join('');
  target.className = 'rewrite-audit';
  target.innerHTML = `<section class="rewrite-audit-head"><div><h3>${escapeHtml(engine === 'engine3' ? 'Engine 3 signal-guided audit' : engine === 'engine2' ? 'Engine 2 independent post-audit' : 'Engine 1 independent post-audit')}</h3><p>${engine === 'engine3' ? 'Engine 3 deliberately uses diagnosed A–I/style statistics. The final detector run is fresh and separate.' : 'This rewrite engine did not receive the detector score as an optimisation target.'}</p></div><strong>${Number(ai.before||0)}% → ${Number(ai.after||0)}%</strong></section><div class="audit-summary"><span>AI Signal ${Number(ai.before||0)}% → ${Number(ai.after||0)}%</span><span>Human-like Style ${Number(human.before||0)}% → ${Number(human.after||0)}%</span><span>Preservation ${data.preservation_certificate?.passed ? 'passed' : 'review'}</span></div><h3>A–I map</h3><div class="audit-grid">${signalRows}</div>${statRows ? `<h3>Statistical map</h3><div class="audit-grid">${statRows}</div>` : ''}`;
}

function developerHeaders() {
  const token = $('developerToken')?.value || sessionStorage.getItem('humanizer_developer_token') || '';
  if ($('developerToken')?.value) sessionStorage.setItem('humanizer_developer_token', $('developerToken').value);
  return token ? {'X-Developer-Token': token} : {};
}

function metricPercent(value) {
  return value === undefined || value === null ? '—' : `${Math.round(Number(value)*100)}%`;
}

function renderValidationCentre(report = {}) {
  const target = $('validationCentre');
  if (!target) return;
  if (!report || report.available === false) {
    target.className = 'validation-centre empty-state';
    target.textContent = report.message || 'No validation report has been generated yet.';
    return;
  }
  const overall = report.overall || {};
  const candidate = report.candidate_metrics || {};
  const candidateCards = Object.entries(candidate).map(([name,m]) => `<article><b>${escapeHtml(labelize(name))}</b><span>AUC ${Number(m.roc_auc||0).toFixed(2)} · F1 ${Number(m.f1||0).toFixed(2)} · FPR ${metricPercent(m.false_positive_rate)}</span></article>`).join('');
  const groupBlock = (title, groups) => {
    const entries = Object.entries(groups || {});
    if (!entries.length) return '';
    return `<details><summary>${escapeHtml(title)}</summary><div class="validation-groups">${entries.map(([name,m])=>`<div><b>${escapeHtml(name)}</b><span>F1 ${Number(m.f1||0).toFixed(2)}</span><span>FPR ${metricPercent(m.false_positive_rate)}</span><span>FNR ${metricPercent(m.false_negative_rate)}</span></div>`).join('')}</div></details>`;
  };
  target.className = 'validation-centre';
  target.innerHTML = `<section class="validation-head"><div><h3>Validation Centre</h3><p>${escapeHtml(report.note || 'Use held-out metrics to judge whether detector changes improve generalisation.')}</p></div><strong>${escapeHtml(report.selected_model || 'active model')}</strong></section><div class="reference-metrics"><span><b>${Number(report.sample_count||0)}</b>Samples</span><span><b>${Number(overall.accuracy||0).toFixed(2)}</b>Accuracy</span><span><b>${Number(overall.precision||0).toFixed(2)}</b>Precision</span><span><b>${Number(overall.recall||0).toFixed(2)}</b>Recall</span><span><b>${Number(overall.f1||0).toFixed(2)}</b>F1</span><span><b>${Number(overall.roc_auc||0).toFixed(2)}</b>ROC-AUC</span><span><b>${metricPercent(overall.false_positive_rate)}</b>FPR</span><span><b>${metricPercent(overall.false_negative_rate)}</b>FNR</span></div>${candidateCards ? `<h3>Held-out candidate comparison</h3><div class="candidate-model-grid">${candidateCards}</div>`:''}${groupBlock('By source/model family',report.by_source_family)}${groupBlock('By discipline',report.by_discipline)}${groupBlock('By document type',report.by_document_type)}${groupBlock('By editing level',report.by_editing_level)}${groupBlock('By word count',report.by_word_count)}`;
}

function renderBenchmarkStatus(payload = {}) {
  const target = $('benchmarkStatus');
  const benchmark = payload.benchmark || payload || {};
  if ($('benchmarkCount')) $('benchmarkCount').textContent = `${Number(benchmark.sample_count||0)} samples`;
  if (!target) return;
  const ext = benchmark.external_detector_disagreement || {};
  target.className = 'benchmark-status';
  target.innerHTML = `<div class="reference-metrics"><span><b>${Number(benchmark.human_count||0)}</b>Human</span><span><b>${Number(benchmark.ai_count||0)}</b>AI/AI-edited</span><span><b>${benchmark.ready_to_train?'Ready':'Not ready'}</b>Training</span><span><b>${ext.mean_score_range == null ? '—' : Number(ext.mean_score_range).toFixed(1)}</b>Mean external detector range</span></div>${benchmark.warning ? `<p class="warning-note">${escapeHtml(benchmark.warning)}</p>`:''}`;
}

async function refreshBenchmark() {
  try {
    const response = await api('/api/developer/benchmark/status',{headers:developerHeaders()});
    const data = await response.json();
    renderBenchmarkStatus(data);
    renderValidationCentre(data.validation || {});
    return data;
  } catch (e) {
    if ($('benchmarkStatus')) { $('benchmarkStatus').className='benchmark-status empty-state'; $('benchmarkStatus').textContent=e.message; }
    return null;
  }
}

async function addBenchmarkSample() {
  const text = $('sourceText')?.value.trim();
  if (!text) return setMessage('Add source text before saving a benchmark sample.','error');
  const external = {};
  [['copyleaks','benchmarkCopyleaks'],['turnitin','benchmarkTurnitin'],['quillbot_scribbr','benchmarkQuillbot']].forEach(([key,id])=>{ const value=$(id)?.value; if(value!=='' && value!=null) external[key]=Number(value); });
  const body = {text, provenance:$('benchmarkProvenance')?.value || 'human_original', source_family:$('benchmarkSourceFamily')?.value || 'unknown', discipline:$('benchmarkDiscipline')?.value || 'unknown', document_type:$('benchmarkDocumentType')?.value || 'unknown', editing_level:$('benchmarkEditingLevel')?.value || 'none', external_scores:external};
  try {
    const response=await api('/api/developer/benchmark/sample',{method:'POST',headers:{'Content-Type':'application/json',...developerHeaders()},body:JSON.stringify(body)});
    const data=await response.json(); renderBenchmarkStatus(data); setMessage('Benchmark sample added with provenance metadata.','success');
  } catch(e) { setMessage(e.message,'error'); }
}

async function trainBenchmark() {
  busy(true,'Extracting benchmark features and selecting the best held-out model…');
  try { const response=await api('/api/developer/benchmark/train',{method:'POST',headers:developerHeaders()}); const data=await response.json(); renderValidationCentre(data.validation||{}); await refreshBenchmark(); activateTab('validation'); setMessage(`Calibration trained. Selected ${data.selected_model || 'model'} from held-out comparison.`,'success'); } catch(e){setMessage(e.message,'error');} finally{busy(false);}
}

async function evaluateBenchmark() {
  busy(true,'Evaluating active model…');
  try { const response=await api('/api/developer/benchmark/evaluate',{method:'POST',headers:developerHeaders()}); const data=await response.json(); renderValidationCentre(data.validation||{}); activateTab('validation'); setMessage('Validation report updated.','success'); } catch(e){setMessage(e.message,'error');} finally{busy(false);}
}

async function runAdversarialAudit() {
  const text=$('sourceText')?.value.trim(); if(!text) return setMessage('Add source text before running a robustness audit.','error');
  busy(true,'Testing detector stability after Engine 1 and Engine 3 editing…');
  try { const response=await api('/api/developer/benchmark/adversarial-audit',{method:'POST',headers:{'Content-Type':'application/json',...developerHeaders()},body:JSON.stringify({text,mode:$('mode')?.value||'deep'})}); const data=await response.json(); if($('rewriteAudit')){$('rewriteAudit').className='rewrite-audit';$('rewriteAudit').innerHTML=`<section class="rewrite-audit-head"><div><h3>Detector robustness audit</h3><p>${escapeHtml(data.note||'')}</p></div><strong>${Number(data.before||0)}%</strong></section><div class="candidate-model-grid"><article><b>Engine 1</b><span>${Number(data.before||0)}% → ${Number(data.engine1?.after||0)}% · preservation ${data.engine1?.preservation?.passed?'pass':'review'}</span></article><article><b>Engine 3</b><span>${Number(data.before||0)}% → ${Number(data.engine3?.after||0)}% · preservation ${data.engine3?.preservation?.passed?'pass':'review'}</span></article></div>`;} activateTab('rewriteaudit'); setMessage('Robustness audit completed.','success'); } catch(e){setMessage(e.message,'error');} finally{busy(false);}
}

function renderPreservation(certificate = null) {
  const target = $('preservationCertificate');
  if (!target) return;
  if (!certificate) { target.className='preservation-certificate empty-state'; target.textContent='A protected-content certificate will appear after humanization.'; return; }
  const checks = certificate.checks || {};
  target.className = `preservation-certificate ${certificate.passed ? 'passed' : 'failed'}`;
  target.innerHTML = `<header><strong>${certificate.passed ? 'Protected-content audit passed' : 'Protected-content audit needs review'}</strong><span>${escapeHtml(certificate.note || '')}</span></header><div>${Object.entries(checks).map(([key,ok])=>`<span class="preservation-chip ${ok?'ok':'bad'}">${ok?'✓':'!'} ${escapeHtml(labelize(key))}</span>`).join('')}</div>`;
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
  busy(true, 'Running calibrated multi-layer AI detection…');
  try {
    const response = await api('/api/analyse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    updateDashboard(await response.json());
    activateTab('detector');
    setMessage('AI detection completed. Review the forensic, statistical, paragraph, calibration and signal-colour evidence together.', 'success');
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
    renderPreservation(data.preservation_certificate || null);
    renderRewriteAudit(data);
    activateTab('revised');
    const engineNote = data.actual_engine === 'engine1_fallback'
      ? ` Engine 2 was unavailable, so Engine 1 fallback was used. ${data.engine_2?.reason || ''}`
      : data.selected_engine === 'engine2'
        ? (data.engine_2?.applied ? ' Engine 2 API rewrite passed preservation and writing-quality checks.' : ` ${data.engine_2?.reason || 'Engine 2 did not apply changes.'}`)
        : data.selected_engine === 'engine3'
          ? ` Engine 3 targeted ${(data.engine_3?.targeted_signals || []).join(', ') || 'no safely editable signal'}; targeted A–I score ${Number(data.engine_3?.targeted_score_before ?? 0)} → ${Number(data.engine_3?.targeted_score_after ?? 0)}.${(data.engine_3?.targeted_statistical_metrics || []).length ? ` Statistical targets: ${(data.engine_3.targeted_statistical_metrics || []).join(', ')}.` : ''}`
          : ' Engine 1 local rewrite completed without an API call.';
    const objectives = data.selected_engine === 'engine3' ? (data.engine_3?.targeted_signals || []) : (data.engine_1?.rewrite_objectives || []);
    const signalNote = objectives.length ? ` Rewrite focus: ${objectives.join(', ')}.` : '';
    const humanLikeNote = ` Human-like style ${Number(humanLikeImprovement.before ?? 0)}% → ${Number(humanLikeImprovement.after ?? 0)}%.`;
    const aiNote = ` Independent post-rewrite AI audit: ${Number(aiImprovement.before ?? data.original_report?.ai_detection_percentage ?? 0)}% → ${Number(aiImprovement.after ?? data.report?.ai_detection_percentage ?? 0)}%.`;
    const outcome = data.changed ? 'Humanisation completed.' : 'No safe rewrite changes were made.';
    setMessage(`${outcome}${engineNote}${signalNote}${aiNote}${humanLikeNote}`, data.changed ? 'success' : '');
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
  if (engineSelect && ['engine1','engine2','engine3'].includes(savedEngine)) engineSelect.value = savedEngine;
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
  if ($('humannessCounter')) $('humannessCounter').textContent='—';
  if ($('decisionStatus')) $('decisionStatus').textContent='—';
  if ($('scoreSource')) $('scoreSource').textContent='—';
  if ($('wordCount')) $('wordCount').textContent='0';
  if ($('sentenceCount')) $('sentenceCount').textContent='0';
  if ($('activeSignals')) $('activeSignals').textContent='0/9';
  if ($('evidenceCount')) $('evidenceCount').textContent='0';
  if ($('proseSegments')) $('proseSegments').textContent='0';
  if ($('flaggedSegments')) $('flaggedSegments').textContent='0';
  if ($('statisticalFingerprint')) $('statisticalFingerprint').textContent='0%';
  if ($('engineSignalTarget')) $('engineSignalTarget').textContent='—';
  if ($('calibrationState')) $('calibrationState').textContent='Fallback';
  if ($('referenceLmState')) $('referenceLmState').textContent='Proxy';
  if ($('highlightedText')) {
    $('highlightedText').textContent='Run AI detection to colour sentences by AI-style signal strength.';
    $('highlightedText').className='document-view empty-state';
  }
  if ($('signalColouredText')) {
    $('signalColouredText').textContent='Run AI detection to colour the text by A–I signal family.';
    $('signalColouredText').className='document-view empty-state';
  }
  renderDetector({});
  renderStatistical([], {});
  renderCalibration({});
  renderSectionProfile({});
  renderRewriteAudit(null);
  renderPreservation(null);
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
$('addBenchmarkBtn')?.addEventListener('click',addBenchmarkSample);
$('refreshBenchmarkBtn')?.addEventListener('click',refreshBenchmark);
$('trainBenchmarkBtn')?.addEventListener('click',trainBenchmark);
$('evaluateBenchmarkBtn')?.addEventListener('click',evaluateBenchmark);
$('adversarialAuditBtn')?.addEventListener('click',runAdversarialAudit);
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>activateTab(tab.dataset.tab)));

fetch('/api/status').then(r=>r.json()).then(data=>{
  if ($('modelStatus')) { const cal=data.calibration?.trained ? `Calibrated on ${Number(data.calibration.sample_count||0)} samples` : 'uncalibrated fallback'; const lm=data.reference_lm?.enabled ? 'reference LM requested' : 'reference LM proxy'; $('modelStatus').textContent=`${data.model.message} · ${cal} · ${lm}`; }
  const labEnabled = Boolean(data.benchmark_lab?.enabled);
  ['validationTab','benchmarkTab'].forEach(id=>{ const el=$(id); if(el) el.hidden=!labEnabled; });
  if (labEnabled) { renderBenchmarkStatus(data.benchmark_lab || {}); const saved=sessionStorage.getItem('humanizer_developer_token'); if(saved && $('developerToken')) $('developerToken').value=saved; }
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
