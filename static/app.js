const $ = (id) => document.getElementById(id);
let currentReport = null;
let currentDocumentId = null;
let currentHumanizeJobId = null;
let currentFormatPreservation = null;

function setMessage(text, kind = '') {
  const el = $('message');
  if (!el) return;
  el.textContent = text;
  el.className = `message ${kind}`;
}

const requestControlIds = [
  'analyseBtn','humanizeBtn','fileInput','docxBtn','annotatedDocxBtn','htmlBtn',
  'addBenchmarkBtn','refreshBenchmarkBtn','trainBenchmarkBtn','evaluateBenchmarkBtn',
  'adversarialAuditBtn','driftBtn','rollbackModelBtn','developerAccessBtn'
];

function busy(state, text = 'Working…') {
  requestControlIds.forEach(id => {
    const el = $(id);
    if (el) el.disabled = state;
  });
  document.querySelectorAll('.promote-model').forEach(el => { el.disabled = state; });
  if (state) setMessage(text);
}

const activityPlans = {
  upload: [
    [8, 'Preparing document…'], [28, 'Sending document…'], [55, 'Reading file…'],
    [76, 'Extracting text…'], [90, 'Preparing Source text…']
  ],
  detect: [
    [8, 'Preparing text…'], [24, 'Screening linguistic signals…'], [46, 'Computing statistical patterns…'],
    [66, 'Scoring prose segments…'], [82, 'Building signal map…'], [92, 'Finalising detector report…']
  ],
  humanize: [
    [8, 'Protecting evidence-bearing content…'], [27, 'Analysing editable prose…'], [48, 'Rewriting scholarly prose…'],
    [70, 'Checking content preservation…'], [85, 'Running independent post-rewrite audit…'], [92, 'Preparing revised text…']
  ],
  export: [
    [12, 'Preparing export…'], [45, 'Generating document…'], [78, 'Packaging file…'], [92, 'Preparing download…']
  ],
  benchmark: [
    [10, 'Preparing benchmark request…'], [38, 'Processing benchmark data…'], [70, 'Updating private validation data…'], [92, 'Finalising…']
  ],
  train: [
    [6, 'Loading benchmark corpus…'], [24, 'Extracting detector features…'], [48, 'Comparing candidate models…'],
    [68, 'Validating held-out performance…'], [84, 'Applying false-positive constraint…'], [93, 'Saving selected model…']
  ],
  evaluate: [
    [10, 'Loading active model…'], [35, 'Scoring held-out samples…'], [66, 'Calculating validation metrics…'], [92, 'Preparing Validation Centre…']
  ],
  audit: [
    [8, 'Preparing robustness test…'], [28, 'Testing original text…'], [50, 'Testing Engine 1 output…'],
    [72, 'Testing Engine 3 output…'], [92, 'Comparing detector stability…']
  ],
  developer: [
    [12, 'Checking developer credentials…'], [40, 'Loading private detector status…'], [70, 'Loading benchmark registry…'], [92, 'Preparing developer tools…']
  ]
};

let activityTimer = null;
let activityHideTimer = null;
let activityProgressValue = 0;
let activityPlan = activityPlans.benchmark;

function activityStageFor(progress) {
  let stage = activityPlan[0]?.[1] || 'Processing request…';
  for (const [threshold, label] of activityPlan) {
    if (progress >= threshold) stage = label;
    else break;
  }
  return stage;
}

function setActivityProgress(progress, stage = null) {
  const wrap = $('activityProgress');
  const ring = $('activityRing');
  const percent = $('activityPercent');
  const stageEl = $('activityStage');
  activityProgressValue = Math.max(0, Math.min(100, Math.round(progress)));
  if (ring) ring.style.setProperty('--activity-angle', `${activityProgressValue * 3.6}deg`);
  if (percent) percent.textContent = `${activityProgressValue}%`;
  if (stageEl) stageEl.textContent = stage || activityStageFor(activityProgressValue);
  if (wrap) wrap.hidden = false;
}

function startActivity(kind, title, initialStage = '') {
  clearInterval(activityTimer);
  clearTimeout(activityHideTimer);
  activityPlan = activityPlans[kind] || activityPlans.benchmark;
  activityProgressValue = 4;
  const wrap = $('activityProgress');
  if (wrap) {
    wrap.hidden = false;
    wrap.className = 'activity-progress active';
  }
  if ($('activityTitle')) $('activityTitle').textContent = title || 'Working';
  setActivityProgress(activityProgressValue, initialStage || activityStageFor(activityProgressValue));
  activityTimer = window.setInterval(() => {
    if (activityProgressValue >= 93) return;
    const step = activityProgressValue < 30 ? 4 : activityProgressValue < 65 ? 2 : 1;
    setActivityProgress(Math.min(93, activityProgressValue + step));
  }, 650);
}

function completeActivity(stage = 'Completed') {
  clearInterval(activityTimer);
  const wrap = $('activityProgress');
  if (wrap) wrap.className = 'activity-progress success';
  setActivityProgress(100, stage);
  activityHideTimer = window.setTimeout(() => {
    if (wrap) wrap.hidden = true;
  }, 1400);
}

function failActivity(stage = 'Request failed') {
  clearInterval(activityTimer);
  const wrap = $('activityProgress');
  if (wrap) wrap.className = 'activity-progress error';
  if ($('activityStage')) $('activityStage').textContent = stage;
  if (wrap) wrap.hidden = false;
  activityHideTimer = window.setTimeout(() => {
    if (wrap) wrap.hidden = true;
  }, 2600);
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
  if ($('wordCount')) $('wordCount').textContent = report.metrics?.word_count ?? 0;
  if ($('sentenceCount')) $('sentenceCount').textContent = report.metrics?.sentence_count ?? 0;
  if ($('activeSignals')) $('activeSignals').textContent = `${report.active_signal_categories ?? 0}/9`;
  if ($('evidenceCount')) $('evidenceCount').textContent = report.signal_evidence_items ?? 0;
  if ($('proseSegments')) $('proseSegments').textContent = report.prose_segment_count ?? detector.segment_count ?? 0;
  if ($('flaggedSegments')) $('flaggedSegments').textContent = report.flagged_prose_segments ?? detector.flagged_segment_count ?? 0;
  if ($('statisticalFingerprint')) $('statisticalFingerprint').textContent = `${Number(report.statistical_fingerprint_percentage ?? detector.statistical_fingerprint_percentage ?? 0)}%`;
  if ($('engineSignalTarget')) { const active=(detector.signals||[]).filter(x=>Number(x.score||0)>0).map(x=>x.key); $('engineSignalTarget').textContent=active.length ? active.join(', ') : 'None'; }
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

  const arithmetic = `<section class="score-arithmetic public-method-note">
    <b>How to read the result</b>
    <span>The headline score combines multiple independent writing-pattern checks.</span>
    <span>Use the sentence map and A–I evidence to inspect the areas that contributed to the result.</span>
    <span>${escapeHtml(detector.decision_status || 'Result available')}: ${escapeHtml(detector.decision_reason || 'Review the highlighted evidence rather than treating the score as proof of authorship.')}</span>
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
  const narrative = `<section class="detector-explanation"><h3>What gave it away</h3><p>${escapeHtml(detector.what_gave_it_away || '')}</p><h3>Interpretation notes</h3><ul>${notes}</ul></section>`;

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
  target.innerHTML = groups.map(group => `
    <section class="concern-group">
      <header><div><h3>${escapeHtml(group.group || '')}</h3><p>${escapeHtml(group.description || '')}</p></div><strong>${Number(group.percentage || 0)}%</strong></header>
      ${(group.metrics || []).map(metric => `<article class="concern-metric">
        <div class="concern-row"><span>${escapeHtml(metric.label || '')}</span><b>${Number(metric.percentage || 0)}%</b></div>
        <div class="bar"><i style="width:${Math.max(0,Math.min(100,Number(metric.percentage||0)))}%"></i></div>
        <ul>${(metric.evidence || []).map(item=>`<li>${escapeHtml(item)}</li>`).join('')}</ul>
      </article>`).join('')}
    </section>`).join('');
}

function renderCalibration(report = {}) {
  const target = $('calibrationDetails');
  if (!target) return;
  const calibration = report.calibration || report.ai_detector?.calibration || {};
  const prediction = report.ai_detector?.calibration_prediction || null;
  const features = report.calibration_features || report.ai_detector?.calibration_features || {};
  const ref = report.reference_lm || report.ai_detector?.reference_lm || {};
  target.className = 'calibration-details';
  const metrics = calibration.metrics || calibration.test_metrics || {};
  const metricCards = calibration.trained ? `<div class="reference-metrics"><span><b>${Number(calibration.sample_count||0)}</b>Training samples</span><span><b>${metrics.false_positive_rate !== undefined ? Math.round(Number(metrics.false_positive_rate)*100)+'%' : '—'}</b>Test FPR</span><span><b>${metrics.false_negative_rate !== undefined ? Math.round(Number(metrics.false_negative_rate)*100)+'%' : '—'}</b>Test FNR</span><span><b>${metrics.f1 !== undefined ? Number(metrics.f1).toFixed(2) : '—'}</b>Test F1</span><span><b>${metrics.roc_auc !== undefined ? Number(metrics.roc_auc).toFixed(2) : '—'}</b>Test ROC-AUC</span></div>` : '';
  const featureRows = Object.entries(features).map(([key,value])=>`<div><span>${escapeHtml(labelize(key))}</span><b>${Number(value).toFixed(2)}</b></div>`).join('');
  const contributions = Object.entries(prediction?.contributions || {}).sort((a,b)=>Math.abs(Number(b[1]))-Math.abs(Number(a[1]))).slice(0,12).map(([key,value])=>`<div><span>${escapeHtml(labelize(key))}</span><b>${Number(value).toFixed(3)}</b></div>`).join('');
  const probabilityBlock = `<section class="calibration-card"><header><div><h3>Private probability diagnostics</h3><p>${escapeHtml(ref.message || 'Reference-language-model diagnostics are disabled or unavailable.')}</p></div><strong>${ref.scored ? 'Reference LM active' : 'Reference LM inactive'}</strong></header>${ref.scored ? `<div class="reference-metrics"><span><b>${Number(ref.perplexity||0).toFixed(2)}</b>Perplexity</span><span><b>${Number(ref.surprisal_mean||0).toFixed(2)}</b>Mean surprisal</span><span><b>${Number(ref.surprisal_std||0).toFixed(2)}</b>Surprisal SD</span><span><b>${Math.round(Number(ref.low_surprisal_share||0)*100)}%</b>Low-surprisal tokens</span><span><b>${Number(ref.curvature_regular_share||0).toFixed(3)}</b>Curvature regularity</span></div>` : ''}</section>`;
  target.innerHTML = `<section class="calibration-card ${calibration.trained ? 'trained' : 'fallback'}"><header><div><h3>${calibration.trained ? `Active learned detector · ${escapeHtml(calibration.model_type || 'model')}` : 'Fallback detector active'}</h3><p>${escapeHtml(calibration.message || '')}</p></div><strong>Private developer view</strong></header>${metricCards}${prediction ? `<p><b>Learned probability:</b> ${Number(prediction.percentage||0)}% · fallback ensemble ${Number(report.ai_detector?.fallback_ensemble_percentage||0)}%</p>` : '<p>No trained benchmark model is installed.</p>'}${contributions ? `<details><summary>Largest feature contributions for this text</summary><div class="calibration-feature-grid">${contributions}</div></details>`:''}<details><summary>Feature vector used by the calibrator</summary><div class="calibration-feature-grid">${featureRows}</div></details></section>${probabilityBlock}`;
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
  target.innerHTML = `<section class="section-profile-summary"><header><div><h3>Section-aware academic profile</h3><p>Section-aware screening allows Methods and Results to be interpreted differently from Discussion or reflective prose.</p></div><strong>Spread ${Number(profile.score_spread||0).toFixed(1)}</strong></header></section><div class="section-profile-grid">${rows}</div>`;
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

function renderRegistry(registry = {}, drift = null) {
  const target=$('modelRegistryStatus');
  if(!target) return;
  const models=registry.models || [];
  const active=registry.active || null;
  const cards=models.slice(0,8).map(item=>`<article class="registry-model ${item.id===active?'active':''}"><div><b>${escapeHtml(item.model_type||'model')}</b><span>${escapeHtml(item.id||'')}</span></div><small>${Number(item.sample_count||0)} samples · threshold ${Number(item.decision_threshold||0.5).toFixed(3)} · FPR ${metricPercent(item.test_metrics?.false_positive_rate)}</small>${item.id===active?'<em>Active</em>':`<button class="secondary promote-model" data-model-id="${escapeHtml(item.id||'')}">Promote</button>`}</article>`).join('');
  const driftBlock=drift?.available ? `<div class="drift-box"><b>Drift: ${escapeHtml(drift.status||'unknown')}</b><span>Index ${Number(drift.drift_index||0).toFixed(2)} from ${Number(drift.recent_samples||0)} recent samples.</span></div>` : `<div class="drift-box"><b>Drift</b><span>${escapeHtml(drift?.message || 'Run a drift check after training and adding recent benchmark samples.')}</span></div>`;
  target.className='benchmark-status';
  target.innerHTML=`<h3>Model registry</h3>${cards || '<p>No trained models are registered yet.</p>'}${driftBlock}`;
  target.querySelectorAll('.promote-model').forEach(btn=>btn.addEventListener('click',()=>promoteModel(btn.dataset.modelId)));
}

async function refreshRegistryAndDrift(showProgress = false) {
  if (showProgress) { busy(true, 'Checking model registry and drift…'); startActivity('benchmark', 'Check detector drift', 'Loading model registry…'); }
  try {
    const [registryResponse,driftResponse]=await Promise.all([
      api('/api/developer/benchmark/registry',{headers:developerHeaders()}),
      api('/api/developer/benchmark/drift',{headers:developerHeaders()})
    ]);
    const registryData=await registryResponse.json();
    const driftData=await driftResponse.json();
    renderRegistry(registryData.registry||{},driftData.drift||{});
    if (showProgress) completeActivity('Registry and drift check completed');
    return {registry:registryData.registry||{},drift:driftData.drift||{}};
  } catch(e) {
    if($('modelRegistryStatus')) {$('modelRegistryStatus').className='benchmark-status empty-state';$('modelRegistryStatus').textContent=e.message;}
    if (showProgress) failActivity('Registry or drift check failed');
    return null;
  } finally {
    if (showProgress) busy(false);
  }
}

async function promoteModel(modelId) {
  if(!modelId) return;
  busy(true, 'Promoting detector model…');
  startActivity('benchmark', 'Promote detector model', 'Applying selected model…');
  try {
    const response=await api('/api/developer/benchmark/promote',{method:'POST',headers:{'Content-Type':'application/json',...developerHeaders()},body:JSON.stringify({model_id:modelId})});
    const data=await response.json();
    renderRegistry(data.registry||{},null);
    setMessage('Selected detector model promoted.','success');
    completeActivity('Detector model promoted');
  } catch(e) { setMessage(e.message,'error'); failActivity('Model promotion failed'); }
  finally { busy(false); }
}

async function rollbackModel() {
  busy(true, 'Rolling back detector model…');
  startActivity('benchmark', 'Rollback detector model', 'Restoring previous model…');
  try {
    const response=await api('/api/developer/benchmark/rollback',{method:'POST',headers:developerHeaders()});
    const data=await response.json();
    renderRegistry(data.registry||{},null);
    setMessage('Detector model rolled back to the previous registered version.','success');
    completeActivity('Previous detector model restored');
  } catch(e) { setMessage(e.message,'error'); failActivity('Model rollback failed'); }
  finally { busy(false); }
}


async function refreshBenchmark(showProgress = false) {
  if (showProgress) { busy(true, 'Refreshing Benchmark Lab…'); startActivity('benchmark', 'Refresh Benchmark Lab', 'Loading benchmark status…'); }
  try {
    const response = await api('/api/developer/benchmark/status',{headers:developerHeaders()});
    const data = await response.json();
    renderBenchmarkStatus(data);
    renderValidationCentre(data.validation || {});
    renderRegistry(data.registry || {}, null);
    if (showProgress) completeActivity('Benchmark status refreshed');
    return data;
  } catch (e) {
    if ($('benchmarkStatus')) { $('benchmarkStatus').className='benchmark-status empty-state'; $('benchmarkStatus').textContent=e.message; }
    if (showProgress) failActivity('Benchmark refresh failed');
    return null;
  } finally {
    if (showProgress) busy(false);
  }
}

async function addBenchmarkSample() {
  const text = $('sourceText')?.value.trim();
  if (!text) return setMessage('Add source text before saving a benchmark sample.','error');
  const external = {};
  [['copyleaks','benchmarkCopyleaks'],['turnitin','benchmarkTurnitin'],['quillbot_scribbr','benchmarkQuillbot']].forEach(([key,id])=>{ const value=$(id)?.value; if(value!=='' && value!=null) external[key]=Number(value); });
  const body = {text, provenance:$('benchmarkProvenance')?.value || 'human_original', source_family:$('benchmarkSourceFamily')?.value || 'unknown', discipline:$('benchmarkDiscipline')?.value || 'unknown', document_type:$('benchmarkDocumentType')?.value || 'unknown', editing_level:$('benchmarkEditingLevel')?.value || 'none', external_scores:external};
  busy(true, 'Adding benchmark sample…');
  startActivity('benchmark', 'Add benchmark sample', 'Saving provenance-labelled text…');
  try {
    const response=await api('/api/developer/benchmark/sample',{method:'POST',headers:{'Content-Type':'application/json',...developerHeaders()},body:JSON.stringify(body)});
    const data=await response.json(); renderBenchmarkStatus(data); setMessage('Benchmark sample added with provenance metadata.','success');
    completeActivity('Benchmark sample saved');
  } catch(e) { setMessage(e.message,'error'); failActivity('Adding benchmark sample failed'); }
  finally { busy(false); }
}

async function trainBenchmark() {
  busy(true,'Extracting benchmark features and selecting the best held-out model…');
  startActivity('train', 'Train/select detector model', 'Loading benchmark corpus…');
  try { const response=await api('/api/developer/benchmark/train',{method:'POST',headers:developerHeaders()}); const data=await response.json(); renderValidationCentre(data.validation||{}); await refreshBenchmark(); await refreshRegistryAndDrift(); activateTab('validation'); setMessage(`Calibration trained. Selected ${data.selected_model || 'model'} from held-out comparison.`,'success'); completeActivity('Training and model selection completed'); } catch(e){setMessage(e.message,'error'); failActivity('Detector training failed');} finally{busy(false);}
}

async function evaluateBenchmark() {
  busy(true,'Evaluating active model…');
  startActivity('evaluate', 'Evaluate detector model', 'Loading active model…');
  try { const response=await api('/api/developer/benchmark/evaluate',{method:'POST',headers:developerHeaders()}); const data=await response.json(); renderValidationCentre(data.validation||{}); activateTab('validation'); setMessage('Validation report updated.','success'); completeActivity('Detector evaluation completed'); } catch(e){setMessage(e.message,'error'); failActivity('Detector evaluation failed');} finally{busy(false);}
}

async function runAdversarialAudit() {
  const text=$('sourceText')?.value.trim(); if(!text) return setMessage('Add source text before running a robustness audit.','error');
  busy(true,'Testing detector stability after Engine 1 and Engine 3 editing…');
  startActivity('audit', 'Run robustness audit', 'Preparing original and rewritten variants…');
  try { const response=await api('/api/developer/benchmark/adversarial-audit',{method:'POST',headers:{'Content-Type':'application/json',...developerHeaders()},body:JSON.stringify({text,mode:$('mode')?.value||'deep'})}); const data=await response.json(); if($('rewriteAudit')){$('rewriteAudit').className='rewrite-audit';$('rewriteAudit').innerHTML=`<section class="rewrite-audit-head"><div><h3>Detector robustness audit</h3><p>${escapeHtml(data.note||'')}</p></div><strong>${Number(data.before||0)}%</strong></section><div class="candidate-model-grid"><article><b>Engine 1</b><span>${Number(data.before||0)}% → ${Number(data.engine1?.after||0)}% · preservation ${data.engine1?.preservation?.passed?'pass':'review'}</span></article><article><b>Engine 3</b><span>${Number(data.before||0)}% → ${Number(data.engine3?.after||0)}% · preservation ${data.engine3?.preservation?.passed?'pass':'review'}</span></article></div>`;} activateTab('rewriteaudit'); setMessage('Robustness audit completed.','success'); completeActivity('Robustness audit completed'); } catch(e){setMessage(e.message,'error'); failActivity('Robustness audit failed');} finally{busy(false);}
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
  busy(true, 'Running multi-layer AI detection…');
  startActivity('detect', 'Detect AI', 'Preparing text for detection…');
  try {
    const response = await api('/api/analyse',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text})});
    updateDashboard(await response.json());
    activateTab('highlight');
    setMessage('AI detection completed. Red areas contain detected AI-style signals; green areas contain no sentence-level signal. Grey areas are protected or excluded structure.', 'success');
    completeActivity('AI detection completed');
  } catch(e) {
    setMessage(e.message,'error');
    failActivity('AI detection failed');
  } finally {
    busy(false);
  }
}

function delay(ms) { return new Promise(resolve => window.setTimeout(resolve, ms)); }

async function waitForHumanizeJob(jobId) {
  const startedAt = Date.now();
  let transientFailures = 0;
  clearInterval(activityTimer);
  while (true) {
    await delay(1200);
    let response;
    try {
      response = await api(`/api/humanize/jobs/${encodeURIComponent(jobId)}`, {method:'GET', cache:'no-store'});
      transientFailures = 0;
    } catch (error) {
      transientFailures += 1;
      if (transientFailures <= 20 && /Failed to fetch|NetworkError|Load failed/i.test(String(error?.message || error))) {
        setActivityProgress(Math.min(94, activityProgressValue), `Connection interrupted. Reconnecting to the active job (${transientFailures}/20)…`);
        await delay(Math.min(10000, 1200 * transientFailures));
        continue;
      }
      throw new Error('The browser could not reconnect to the active humanization job. The server job may still be running; refresh once connectivity is restored.');
    }
    const job = await response.json();
    let stage = job.stage || 'Humanization in progress…';
    const checkpoint = job.checkpoint_summary || {};
    if (checkpoint.total_batches) {
      stage = `Humanizing document: ${Number(checkpoint.completed_batches || 0)}/${Number(checkpoint.total_batches)} batches; ${Number(checkpoint.completed_words || 0).toLocaleString()}/${Number(checkpoint.total_words || 0).toLocaleString()} words processed.`;
    } else if (Date.now() - startedAt > 20 * 60 * 1000 && job.status === 'running') {
      stage = `${stage} Large-document processing is continuing on the server.`;
    }
    setActivityProgress(Number(job.progress ?? activityProgressValue), stage);
    if (job.status === 'completed') return job.result;
    if (job.status === 'failed') throw new Error(job.error || 'Humanization job failed.');
  }
}

async function humanize() {
  const text = $('sourceText')?.value.trim();
  if (!text) return setMessage('Add text before humanising.', 'error');
  busy(true, 'Applying protected scholarly refinement…');
  startActivity('humanize', 'Humanize scholarly text', 'Submitting protected rewrite job…');
  try {
    const startResponse = await api('/api/humanize/jobs',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,mode:$('mode')?.value || 'balanced',engine:$('engine')?.value || 'engine1',engine2_model:$('engine2Model')?.value || 'v2',document_id:currentDocumentId})});
    const started = await startResponse.json();
    if (!started.job_id) throw new Error('Humanization job could not be started.');
    setActivityProgress(Number(started.progress ?? 2), started.stage || 'Queued for humanization…');
    const data = await waitForHumanizeJob(started.job_id);
    currentHumanizeJobId = data.humanize_job_id || started.job_id || null;
    if (data.document_id) currentDocumentId = data.document_id;
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
    const formatNote = data.format_preserving_export
      ? ` Word format preservation passed. ${Number(data.document_structure?.tables||0)} table(s) and the original document structure remain intact; ${Number(data.document_structure?.changed_paragraphs||0)} prose paragraph(s) were patched in place.`
      : '';
    setMessage(`${outcome}${engineNote}${signalNote}${aiNote}${humanLikeNote}${formatNote}`, data.changed ? 'success' : '');
    completeActivity(data.changed ? 'Humanization completed' : 'Completed with no safe changes');
  } catch(e) {
    setMessage(e.message,'error');
    failActivity('Humanization failed');
  } finally {
    busy(false);
  }
}

async function uploadFile(file) {
  if (!file) return;
  busy(true,'Extracting document text…');
  startActivity('upload', 'Upload document', `Preparing ${file.name}…`);
  const form = new FormData();
  form.append('file',file,file.name);
  try {
    const response=await api('/api/upload',{method:'POST',body:form});
    const data=await response.json();
    if (!data.text || !String(data.text).trim()) throw new Error('The document was read but no usable text was extracted.');
    if ($('sourceText')) {
      $('sourceText').value=data.text;
      $('sourceText').focus();
      $('sourceText').scrollTop=0;
    }
    if ($('revisedText')) $('revisedText').value='';
    currentReport=null;
    currentDocumentId = data.document_id || null;
    currentHumanizeJobId = null;
    currentFormatPreservation = data.format_preservation || null;
    activateTab('detector');
    const formatNote = data.format_preservation?.available
      ? ` Format-preserving Word mode is active: ${Number(data.format_preservation.tables||0)} table(s) and ${Number(data.format_preservation.locked_paragraphs||0)} structurally sensitive paragraph(s) will remain in the original DOCX package.`
      : '';
    setMessage(`${data.filename || file.name} extracted to Source text. Click Detect AI when ready.${formatNote}`, 'success');
    completeActivity('Text extracted to Source text');
  } catch(e) {
    setMessage(`Upload failed: ${e.message}`,'error');
    failActivity('Document upload or extraction failed');
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
  busy(true, 'Preparing export…');
  startActivity('export', 'Export file', `Preparing ${filename}…`);
  try {
    const response=await api(url,{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,title:'Scholarly Humanizer Output',annotated})});
    const blob=await response.blob();
    const link=document.createElement('a');
    link.href=URL.createObjectURL(blob);
    link.download=filename;
    link.click();
    URL.revokeObjectURL(link.href);
    setMessage(`${filename} exported.`, 'success');
    completeActivity('Export ready');
  } catch(e) {
    setMessage(e.message,'error');
    failActivity('Export failed');
  } finally {
    busy(false);
  }
}

async function exportHumanizedWord() {
  const text = ($('revisedText')?.value || '').trim();
  if (!text) return setMessage('Humanize the source text first, then export the revised text to Word.','error');
  busy(true, 'Preparing Word export…');
  startActivity('export', 'Export humanized Word', 'Preparing revised text…');
  try {
    const response=await api('/api/export/docx',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({text,title:'',annotated:false,document_id:currentDocumentId,humanize_job_id:currentHumanizeJobId})});
    const blob=await response.blob();
    const link=document.createElement('a');
    link.href=URL.createObjectURL(blob);
    const disposition = response.headers.get('Content-Disposition') || '';
    const filenameMatch = disposition.match(/filename="?([^";]+)"?/i);
    link.download = filenameMatch ? filenameMatch[1] : 'humanized_scholarly_text.docx';
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(()=>URL.revokeObjectURL(link.href),1000);
    setMessage('Humanized text exported to Word (.docx).','success');
    completeActivity('Word document ready');
  } catch(e) {
    setMessage(e.message,'error');
    failActivity('Word export failed');
  } finally {
    busy(false);
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
      ? 'Choose V1 (Light) for lighter API refinement or V2 (Moderate) for stronger API refinement.'
      : 'Engine 2 is not enabled on this deployment. Ask the administrator to configure API rewriting.';
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
  if (engine2ModelSelect && ['v1','v2'].includes(savedModel)) engine2ModelSelect.value = savedModel;
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
  currentDocumentId=null;
  currentHumanizeJobId=null;
  currentFormatPreservation=null;
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
  if ($('wordCount')) $('wordCount').textContent='0';
  if ($('sentenceCount')) $('sentenceCount').textContent='0';
  if ($('activeSignals')) $('activeSignals').textContent='0/9';
  if ($('evidenceCount')) $('evidenceCount').textContent='0';
  if ($('proseSegments')) $('proseSegments').textContent='0';
  if ($('flaggedSegments')) $('flaggedSegments').textContent='0';
  if ($('statisticalFingerprint')) $('statisticalFingerprint').textContent='0%';
  if ($('engineSignalTarget')) $('engineSignalTarget').textContent='—';
  if ($('highlightedText')) {
    $('highlightedText').textContent='Run AI detection to colour AI-signal areas red and no-signal areas green.';
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
$('docxBtn')?.addEventListener('click',()=>exportHumanizedWord());
$('annotatedDocxBtn')?.addEventListener('click',()=>exportFile('/api/export/docx',true,'ai_signal_diagnostic.docx'));
$('htmlBtn')?.addEventListener('click',()=>exportFile('/api/export/html',true,'ai_signal_diagnostic.html'));
$('addBenchmarkBtn')?.addEventListener('click',addBenchmarkSample);
$('refreshBenchmarkBtn')?.addEventListener('click',()=>refreshBenchmark(true));
$('trainBenchmarkBtn')?.addEventListener('click',trainBenchmark);
$('evaluateBenchmarkBtn')?.addEventListener('click',evaluateBenchmark);
$('adversarialAuditBtn')?.addEventListener('click',runAdversarialAudit);
$('driftBtn')?.addEventListener('click',()=>refreshRegistryAndDrift(true));
$('rollbackModelBtn')?.addEventListener('click',rollbackModel);
document.querySelectorAll('.tab').forEach(tab=>tab.addEventListener('click',()=>activateTab(tab.dataset.tab)));

async function unlockDeveloper() {
  const stored=sessionStorage.getItem('humanizer_developer_token') || '';
  const token=window.prompt('Developer password', stored);
  if (token===null) return;
  sessionStorage.setItem('humanizer_developer_token',token);
  if ($('developerToken')) $('developerToken').value=token;
  busy(true, 'Unlocking developer tools…');
  startActivity('developer', 'Developer access', 'Checking developer password…');
  try {
    const headers={'X-Developer-Token':token};
    const response=await api('/api/developer/detector/status',{headers});
    const data=await response.json();
    ['calibrationTab','validationTab','benchmarkTab'].forEach(id=>{const el=$(id);if(el)el.hidden=false;});
    renderBenchmarkStatus(data.benchmark || {});
    renderValidationCentre(data.validation || {});
    renderRegistry(data.registry || {}, null);
    await refreshRegistryAndDrift();
    const text=$('sourceText')?.value.trim();
    if (text) {
      const analysis=await api('/api/developer/analyse',{method:'POST',headers:{'Content-Type':'application/json',...headers},body:JSON.stringify({text})});
      const privateReport=await analysis.json();
      renderCalibration(privateReport);
    } else {
      renderCalibration({calibration:data.calibration,reference_lm:data.reference_lm,decision_status:'Private developer view'});
    }
    setMessage('Developer tools unlocked for this browser session.','success');
    completeActivity('Developer tools unlocked');
  } catch(e) {
    sessionStorage.removeItem('humanizer_developer_token');
    if ($('developerToken')) $('developerToken').value='';
    setMessage(`Developer access failed: ${e.message}`,'error');
    failActivity('Developer access failed');
  } finally {
    busy(false);
  }
}

$('developerAccessBtn')?.addEventListener('click',unlockDeveloper);

fetch('/api/status').then(r=>r.json()).then(data=>{
  const developerButton=$('developerAccessBtn');
  if (developerButton) developerButton.hidden=!Boolean(data.developer_lab_available);
  const engine2 = data.engines?.engine2;
  engine2Configured = Boolean(engine2?.configured);
  const opt = $('engine2Option');
  if (opt) {
    opt.disabled = false;
    opt.textContent = 'Engine 2, API rewrite';
  }
  syncEngineControls();
}).catch(()=>{
  engine2Configured = false;
  const opt = $('engine2Option');
  if (opt) { opt.disabled = false; opt.textContent='Engine 2, API rewrite'; }
  syncEngineControls();
});
