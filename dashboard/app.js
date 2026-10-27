(() => {
const $ = (id) => document.getElementById(id);
const local = ['localhost', '127.0.0.1'].includes(location.hostname);
let api = local ? `${location.origin}/api/v1` : '';
try { api = localStorage.getItem('biomedical-blue-api') || api; } catch {}
if (local && (!api || api === 'http://127.0.0.1:8000/api/v1' || api === 'http://localhost:8000/api/v1')) api = `${location.origin}/api/v1`;
const emptyMarkup = $('result-body').innerHTML;
let activeRequest = null;
let previewUrl = null;
let toastTimer;
const label = v => ({ SUPPORTED: 'Supported', REFUTED: 'Refuted', NEI: 'Not enough information', NOT_ENOUGH_INFORMATION: 'Not enough information' }[v] || 'No verdict available');
function text(tag, value, className) { const e = document.createElement(tag); e.textContent = value; if (className) e.className = className; return e; }
function showError(message) { $('error').textContent = message; $('error').hidden = !message; }
function notify(message) { clearTimeout(toastTimer); $('toast').textContent = message; $('toast').hidden = false; toastTimer = setTimeout(() => $('toast').hidden = true, 4000); }
function percent(value) { return typeof value === 'number' && Number.isFinite(value) && value >= 0 && value <= 1 ? `${(value * 100).toFixed(1)}%` : 'Not reported'; }
function imageStatus(phase) {
 if (!phase?.has_image) return null;
 const processed = phase.status === 'SUCCESS';
 const block = text('div', '', 'image-analysis');
 block.append(text('strong', processed ? 'Image processed' : 'Image analysis incomplete'),
  text('p', processed ? 'An experimental image/text representation was computed. The claim verdict uses text and retrieved research; this image result is not a clinical diagnosis.' : 'The image branch did not complete. The image has no role in this verdict.', 'result-footnote'));
 return block;
}
$('claim').addEventListener('input', () => { $('count').textContent = `${$('claim').value.length.toLocaleString()} / 5,000`; });
document.querySelectorAll('[data-claim]').forEach(b => b.addEventListener('click', () => { $('claim').value = b.dataset.claim; $('claim').dispatchEvent(new Event('input')); $('claim').focus(); }));
document.addEventListener('keydown', event => { if ((event.metaKey || event.ctrlKey) && event.key === 'Enter' && !activeRequest) { event.preventDefault(); $('claim-form').requestSubmit(); } });
$('upload-button').onclick = () => $('image').click();
function clearFile() { $('image').value = ''; $('attachment').hidden = true; if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = null; $('image-preview').removeAttribute('src'); }
$('remove-image').onclick = clearFile;
$('image').onchange = () => { const f = $('image').files[0]; if (!f) return; if (!['image/png', 'image/jpeg', 'image/webp'].includes(f.type)) { clearFile(); showError('Choose a PNG, JPEG, or WebP image.'); return; } if (f.size > 10 * 1024 * 1024) { clearFile(); showError('Choose an image smaller than 10 MB.'); return; } if (previewUrl) URL.revokeObjectURL(previewUrl); previewUrl = URL.createObjectURL(f); $('image-preview').src = previewUrl; $('file-name').textContent = f.name; $('attachment').hidden = false; showError(''); };
function validateEvidenceResult(data) {
 const direction = { REFUTED: 'refuted', SUPPORTED: 'supported' }[data.final_verdict];
 if (!direction) return;
 const review = data.phase4?.model?.metadata?.relation_review;
 if (!review || review.verdict !== data.final_verdict) throw Error('Evidence validation failed: this decisive result has not passed the current relation review. Please run the analysis again.');
 const pairs = data.phase4?.pairwise_predictions;
 if (!Array.isArray(pairs) || !pairs.some(p => {
  const v = p.probabilities;
  return v && ['refuted', 'nei', 'supported'].every(k => Number.isFinite(v[k]) && v[k] >= 0 && v[k] <= 1) &&
   ['refuted', 'nei', 'supported'].filter(k => k !== direction).every(k => v[direction] > v[k]);
 })) throw Error('Evidence validation failed: the retrieved passages do not justify this verdict. No verified conclusion is available.');
}
function renderUnverified(message, reason, diagnostics, elapsedMs) {
 const body = $('result-body'); body.replaceChildren();
 const panel = text('div', '', 'result-content');
 panel.append(text('span', 'EVIDENCE CHECK', 'eyebrow'), text('h2', 'Not enough information to accept a verdict', 'verdict'),
  text('p', $('claim').value, 'result-claim'), text('p', message),
  text('p', 'No supported or refuted conclusion was accepted. This is a verification failure, not a finding that the claim is true or false.', 'result-footnote'));
 const modelScores = diagnostics?.proposed_scores;
 const stats = text('div', '', 'result-stats');
 const proposed = diagnostics?.proposed_verdict;
 const sourceCount = diagnostics?.retrieval?.selected_documents ?? (Array.isArray(diagnostics?.evidence) ? diagnostics.evidence.length : null);
 for (const [value, name] of [[sourceCount === null ? '—' : String(sourceCount), 'Sources'],
  [percent(modelScores?.[proposed]), `Proposed ${label(proposed)} score · rejected`],
  [Number.isFinite(elapsedMs) ? `${(elapsedMs / 1000).toFixed(1)}s` : '—', 'Request time']]) {
  const item = text('div', ''); item.append(text('strong', value), text('span', name)); stats.append(item);
 }
 panel.append(stats);
 const imageNote = imageStatus(diagnostics?.image_analysis); if (imageNote) panel.append(imageNote);
 if (modelScores) {
  const distribution = text('div', '', 'distribution');
  distribution.append(text('h3', 'Model scores before evidence validation'), text('p', 'No class was accepted. The classifier proposed the highest-scoring class, but the evidence review rejected it. These are the original model scores, not a confidence score for the withheld outcome.', 'result-footnote'));
  for (const [key, title, kind] of [['SUPPORTED', 'Supported', 'supported'], ['REFUTED', 'Refuted', 'refuted'], ['NEI', 'Not enough information', 'nei']]) {
   const value = modelScores[key]; const row = text('div', '', 'probability'); const heading = text('div', '', 'score-line');
   heading.append(text('span', title), text('span', percent(value))); row.append(heading);
   if (percent(value) !== 'Not reported') { const bar = text('div', '', 'meter'); const fill = text('div', ''); fill.style.width = `${value * 100}%`; fill.dataset.type = kind; bar.append(fill); row.append(bar); }
   distribution.append(row);
  }
  panel.append(distribution);
 }
 if (reason) { const details = text('details', '', 'verification-notes'); details.append(text('summary', 'Why verification failed'), text('p', reason)); panel.append(details); }
 if (diagnostics) {
  panel.append(text('h3', 'What happened'), text('p', `Stopped at: ${diagnostics.failure_stage || 'Evidence validation'}`, 'result-footnote'), text('p', diagnostics.explanation || reason || message),
   text('p', diagnostics.scientific_meaning || '', 'result-footnote'));
  const retrieval = diagnostics.retrieval || {};
  panel.append(text('p', `${retrieval.selected_documents ?? '—'} selected documents · ${retrieval.corpus_documents ?? '—'} local corpus records · ${retrieval.live_search ? 'Live retrieval used' : 'No live search used'}`, 'result-footnote'));
  const rejected = text('details', '', 'verification-notes');
  rejected.append(text('summary', 'Rejected model proposal — diagnostic only'), text('p', `Proposed class: ${label(diagnostics.proposed_verdict)}. This was rejected and is not the result.`));
  if (diagnostics.proposed_scores) for (const [key,value] of Object.entries(diagnostics.proposed_scores)) rejected.append(text('p', `${label(key)}: ${percent(value)}`, 'result-footnote'));
  panel.append(rejected);
  const review = diagnostics.relation_review;
  if (review) {
   panel.append(text('h3', 'Independent reviewer disagreement'), text('p', `Reviewer assessment: ${label(review.verdict)} — not an accepted final verdict.`, 'result-footnote'), text('p', review.reason));
   for (const cite of review.citations || []) panel.append(text('blockquote', `${cite.quote} [${cite.evidence_id}]`));
   panel.append(text('p', 'The reviewer is also a fallible model. Its disagreement with the classifier prevented an accepted verdict.', 'result-footnote'));
  }
  panel.append(text('h3', 'Evidence retrieved'));
  for (const item of diagnostics.evidence || []) {
   const source = text('details', '', 'scored-passage'); source.open = true;
   source.append(text('summary', item.title || item.evidence_id), text('p', item.text || 'No passage available'));
   const scores = item.probabilities || {};
   source.append(text('p', `Passage prediction: ${label(item.pairwise_label)} · Refuted ${percent(scores.refuted)} · Neutral ${percent(scores.nei)} · Supported ${percent(scores.supported)}`, 'result-footnote'));
   source.append(text('p', `Source: ${item.source || 'Unknown'} · ID: ${item.evidence_id} · Relevance score: ${percent(item.relevance)}`, 'result-footnote'));
   if (item.source_url && /^https:\/\/pubmed\.ncbi\.nlm\.nih\.gov\/\d+\/$/.test(item.source_url)) {
    const link = text('a', 'Open original paper ↗'); link.href = item.source_url; link.target = '_blank'; link.rel = 'noopener noreferrer'; source.append(link);
   }
   panel.append(source);
  }
  for (const note of diagnostics.limitations || []) panel.append(text('p', note, 'result-footnote'));
  const exportButton = text('button', 'Export diagnostic report'); exportButton.type = 'button';
  exportButton.onclick = () => { const url = URL.createObjectURL(new Blob([JSON.stringify({status:'UNVERIFIED',request_elapsed_ms:elapsedMs,diagnostics},null,2)],{type:'application/json'})); const a = document.createElement('a'); a.href = url; a.download = 'unverified-diagnostic.json'; a.click(); setTimeout(()=>URL.revokeObjectURL(url),1000); }; panel.append(exportButton);
 }
 body.append(panel); $('result-tag').textContent = 'Verdict withheld'; $('analysis-state').textContent = 'Evidence check failed';
 $('sample-button').textContent = 'Clear result'; $('sample-button').onclick = reset;
}
function render(data, sample = false) {
 if (!sample) validateEvidenceResult(data);
 const body = $('result-body'); body.replaceChildren(); const c = text('div', '', 'result-content'); body.append(c);
 if (sample) c.append(text('p', 'INTERACTIVE SAMPLE · Illustrative data, not a medical finding.', 'sample-note'));
 c.append(text('span', sample ? 'SAMPLE VERDICT' : 'VERIFICATION RESULT', 'eyebrow'));
 const verdict = text('h2', label(data.final_user_facing_verdict || data.final_verdict), 'verdict'); verdict.dataset.verdict = data.final_verdict; c.append(verdict, text('p', data.request?.claim || '', 'result-claim'));
 const evidence = Array.isArray(data.evidence) ? data.evidence : [];
 const stats = text('div', '', 'result-stats');
 for (const [value, name] of [[String(evidence.length), 'Sources'], [percent(data.final_verifier_score), 'Verifier score'], [Number.isFinite(data.total_latency_ms) ? `${(data.total_latency_ms / 1000).toFixed(1)}s` : '—', 'Analysis time']]) { const item = text('div', ''); item.append(text('strong', value), text('span', name)); stats.append(item); } c.append(stats);
 const imageNote = imageStatus(data.phase1); if (imageNote) c.append(imageNote);
 const tabs = text('div', '', 'result-tabs'); tabs.setAttribute('role', 'tablist'); tabs.setAttribute('aria-label', 'Analysis details'); c.append(tabs);
 const panels = [];
 ['Overview', `Evidence (${evidence.length})`, 'Pipeline', 'Insights'].forEach((name, i) => {
  const button = text('button', name); button.type = 'button'; button.id = `result-tab-${i}`; button.setAttribute('role', 'tab'); button.setAttribute('aria-controls', `result-panel-${i}`); button.setAttribute('aria-selected', String(i === 0)); button.tabIndex = i === 0 ? 0 : -1;
  const panel = text('section', '', 'result-panel'); panel.id = `result-panel-${i}`; panel.setAttribute('role', 'tabpanel'); panel.setAttribute('aria-labelledby', button.id); panel.tabIndex = 0; panel.hidden = i !== 0; panels.push(panel);
  button.onclick = () => { [...tabs.children].forEach((b, j) => { b.setAttribute('aria-selected', String(j === i)); b.tabIndex = j === i ? 0 : -1; panels[j].hidden = j !== i; }); };
  button.onkeydown = e => { let next; if (e.key === 'ArrowRight') next = (i + 1) % 4; if (e.key === 'ArrowLeft') next = (i + 3) % 4; if (e.key === 'Home') next = 0; if (e.key === 'End') next = 3; if (next !== undefined) { e.preventDefault(); tabs.children[next].click(); tabs.children[next].focus(); } }; tabs.append(button);
 }); c.append(...panels);
 const explanation = data.phase5?.explanation_status === 'ACCEPTED' ? data.phase5?.explanation : null;
 panels[0].append(text('p', explanation || (data.phase5?.explanation_status === 'QUARANTINED' ? 'The explanation did not pass grounding checks and has been withheld.' : 'No accepted explanation is available.'), 'explanation'));
 if (!sample && data.provenance?.mode === 'enhanced') panels[0].append(text('p', 'Biomedical NLI · Enhanced research mode', 'result-footnote'));
 const probs = data.phase4?.probabilities;
 if (probs) { const distribution = text('div', '', 'distribution'); distribution.append(text('h3', 'How the model weighed the claim')); for (const [key, title] of [['supported', 'Supported'], ['refuted', 'Refuted'], ['nei', 'Insufficient evidence']]) { const value = probs[key]; const row = text('div', '', 'probability'); const heading = text('div', '', 'score-line'); heading.append(text('span', title), text('span', percent(value))); row.append(heading); if (percent(value) !== 'Not reported') { const bar = text('div', '', 'meter'); const fill = text('div', ''); fill.style.width = `${value * 100}%`; fill.dataset.type = key; bar.append(fill); row.append(bar); } distribution.append(row); } panels[0].append(distribution); }
 if (!sample && data.phase4?.warnings?.length) { const notes = text('details', '', 'verification-notes'); notes.append(text('summary', 'Verification limits & notes')); data.phase4.warnings.forEach(w => notes.append(text('p', w, 'result-footnote'))); panels[0].append(notes); }
 panels[0].append(text('p', 'Model scores express the verifier’s output, not clinical certainty.', 'result-footnote'));
 const search = document.createElement('input'); search.type = 'search'; search.placeholder = 'Search titles, sources, or evidence…'; search.className = 'evidence-search'; search.setAttribute('aria-label', 'Search evidence');
 const list = text('div', '', 'evidence-list'); const matches = text('p', '', 'result-footnote'); matches.setAttribute('role', 'status');
 function showEvidence() { list.replaceChildren(); const query = search.value.trim().toLowerCase(); const filtered = evidence.filter(e => `${e.title || ''} ${e.source || ''} ${e.text || ''}`.toLowerCase().includes(query)); matches.textContent = `${filtered.length} of ${evidence.length} sources`; if (!filtered.length) list.append(text('p', evidence.length ? 'No sources match this search.' : 'No evidence was returned.', 'explanation')); filtered.forEach((e, i) => { const d = document.createElement('details'); const summary = text('summary', ''); summary.append(text('span', String(i + 1).padStart(2, '0'), 'source-number'), text('span', e.title || e.evidence_id || 'Evidence')); d.append(summary, text('p', e.text || 'No excerpt available.')); const metadata = text('div', '', 'source-meta'); metadata.append(text('span', `Source: ${e.source || 'Not provided'}`), text('span', `Authority: ${e.authority_tier || 'Not reported'}`)); if (e.publication_date) metadata.append(text('span', `Published: ${e.publication_date}`)); if (e.retrieval_origin === 'live_pubmed') metadata.append(text('span', 'Retrieved from PubMed')); if (e.source_url && /^https:\/\/pubmed\.ncbi\.nlm\.nih\.gov\/\d+\/$/.test(e.source_url)) { const link = text('a', 'Open in PubMed ↗'); link.href = e.source_url; link.target = '_blank'; link.rel = 'noopener noreferrer'; metadata.append(link); } if (e.lineage?.used_by_phase4) metadata.append(text('span', 'Used in verdict')); if (e.lineage?.cited_by_phase5) metadata.append(text('span', 'Cited in explanation')); const pair = data.phase4?.pairwise_predictions?.find(p => p.evidence_id === e.evidence_id); if (pair?.scored_text) { const excerpt = text('details', '', 'scored-passage'); excerpt.append(text('summary', 'Passage scored by the verifier'), text('p', pair.scored_text)); d.append(excerpt); } d.append(metadata); list.append(d); }); }
 search.oninput = showEvidence; panels[1].append(search, matches, list); showEvidence();
 const phaseNames = ['Represent', 'Retrieve', 'Rank', 'Verify', 'Explain'];
 for (let i = 1; i <= 5; i++) { const p = data[`phase${i}`]; const row = text('div', '', 'trace-row'); row.append(text('span', String(i).padStart(2, '0'), 'trace-number')); const info = text('div', ''); info.append(text('strong', phaseNames[i - 1]), text('span', p?.status ? p.status.replaceAll('_', ' ').toLowerCase() : 'not reported')); if (p?.error?.message) info.append(text('p', p.error.message, 'error')); if (Array.isArray(p?.warnings)) p.warnings.forEach(w => info.append(text('p', w, 'trace-warning'))); row.append(info, text('span', Number.isFinite(p?.timing?.latency_ms) ? `${(p.timing.latency_ms / 1000).toFixed(2)}s` : '—', 'trace-time')); panels[2].append(row); }
 panels[3].append(buildInsights(data, evidence));
 const actions = text('div', '', 'result-actions'); const copy = text('button', 'Copy summary'); copy.type = 'button'; copy.onclick = async () => { try { await navigator.clipboard.writeText(`${sample ? 'ILLUSTRATIVE SAMPLE\n' : ''}${data.request?.claim || ''}\n\nVerdict: ${label(data.final_verdict)}\nVerifier score: ${percent(data.final_verifier_score)}\n\n${explanation || 'No accepted explanation available.'}`); notify('Summary copied.'); } catch { notify('Clipboard access is unavailable in this browser.'); } };
 const download = text('button', 'Export JSON'); download.type = 'button'; download.onclick = () => { const url = URL.createObjectURL(new Blob([JSON.stringify({ illustrative_sample: sample, ...data }, null, 2)], { type: 'application/json' })); const a = document.createElement('a'); a.href = url; a.download = sample ? 'biomedical-sample.json' : 'biomedical-analysis.json'; a.click(); setTimeout(() => URL.revokeObjectURL(url), 1000); }; actions.append(copy, download); c.append(actions);
 $('result-tag').textContent = sample ? 'Sample' : 'Result'; $('analysis-state').textContent = sample ? 'Illustrative sample only' : 'Analysis complete'; $('sample-button').textContent = 'Clear result'; $('sample-button').onclick = reset;
}
function reset() { $('result-body').innerHTML = emptyMarkup; $('result-tag').textContent = 'Preview'; $('analysis-state').textContent = 'No analysis run yet'; $('sample-button').textContent = 'Explore sample'; $('sample-button').onclick = sample; }
function sample() { render({ request: { claim: 'An example biomedical claim would appear here.' }, final_verdict: 'NEI', final_verifier_score: 0.62, total_latency_ms: 12400, phase1: { status: 'SUCCESS', timing: { latency_ms: 800 } }, phase2: { status: 'SUCCESS', timing: { latency_ms: 4100 } }, phase3: { status: 'SUCCESS', timing: { latency_ms: 1500 } }, phase4: { status: 'SUCCESS', timing: { latency_ms: 3800 }, probabilities: { supported: 0.23, refuted: 0.15, nei: 0.62 } }, phase5: { status: 'SUCCESS', timing: { latency_ms: 2200 }, explanation_status: 'ACCEPTED', explanation: 'This is an interface demonstration. A real result connects the verdict to retrieved research, explains what the evidence supports, and makes uncertainty visible. Open Evidence to inspect sources, or Pipeline to review each stage.' }, evidence: [{ evidence_id: 'sample-01', title: 'Example systematic review', text: 'Illustrative excerpt. In a real analysis, this area contains the retrieved passage used to assess the claim. No scientific finding is asserted by this sample.', source: 'Sample research collection', authority_tier: 'A', relevance_score: 0.86, combined_score: 0.81, hierarchy_weight: 1, lineage: { used_by_phase4: true, cited_by_phase5: true } }, { evidence_id: 'sample-02', title: 'Example observational study', text: 'Illustrative excerpt showing a second source. Authority tiers and timing values in this sample are demonstration data only.', source: 'Sample research collection', authority_tier: 'C', relevance_score: 0.71, combined_score: 0.43, hierarchy_weight: 0.6, lineage: { used_by_phase4: true, cited_by_phase5: false } }] }, true); }
$('sample-button').onclick = sample;
$('claim-form').onsubmit = async event => {
 event.preventDefault(); if (activeRequest) return; showError(''); if (!$('claim').value.trim()) { showError('Enter a biomedical claim first.'); return; } if (!api) { showError('The backend is unavailable. Check that the local project launcher is running.'); return; }
 const form = new FormData(); form.append('request_id', crypto.randomUUID()); form.append('claim', $('claim').value.trim()); if ($('image').files[0]) form.append('image', $('image').files[0]);
 const controls = [...$('claim-form').querySelectorAll('input,textarea,button'), ...document.querySelectorAll('[data-claim]'), $('sample-button')]; controls.forEach(c => c.disabled = true);
 const controller = new AbortController(); activeRequest = controller; let timedOut = false; const timeout = setTimeout(() => { timedOut = true; controller.abort(); }, 300000); const started = Date.now();
 $('submit').textContent = 'Analyzing…'; $('result-tag').textContent = 'Running'; $('analysis-state').textContent = 'Waiting for the pipeline'; $('result-body').innerHTML = '<div class="loading" role="status"><div class="spinner"></div><h2>Following the evidence.</h2><p>Waiting for the completed analysis.<br>Stage-level progress is not available.</p><span id="elapsed">0s elapsed</span><button type="button" id="cancel-request">Cancel request</button></div>';
 const elapsed = setInterval(() => { if ($('elapsed')) $('elapsed').textContent = `${Math.floor((Date.now() - started) / 1000)}s elapsed`; }, 1000); $('cancel-request').onclick = () => controller.abort();
 try { const response = await fetch(`${api}/inferences`, { method: 'POST', headers: { Accept: 'application/json' }, body: form, signal: controller.signal }); let d; try { d = await response.json(); } catch { throw Error(`The backend returned an unreadable response (${response.status}).`); } if (!response.ok) { const failure = Error(typeof d.detail === 'string' ? d.detail : d.detail?.message || `Analysis failed (${response.status}).`); failure.code = d.detail?.error; failure.reason = d.detail?.reason; failure.diagnostics = d.detail?.diagnostics; throw failure; } if (!d.request || !d.phase4 || !d.final_verdict) throw Error('The backend returned an unexpected result format.'); render(d); }
 catch (e) { if (e.code === 'EVIDENCE_VALIDATION_FAILED' || e.message?.startsWith('Evidence validation failed:')) { renderUnverified(e.message, e.reason, e.diagnostics, Date.now() - started); return; } reset(); if (controller.signal.aborted && !timedOut) notify('Request canceled. The backend may still finish processing.'); else showError(timedOut ? 'The request timed out after five minutes. Check the backend before retrying.' : e instanceof TypeError ? 'Could not reach the backend. Check that it is running and allows this site’s address.' : e.message); }
 finally { clearTimeout(timeout); clearInterval(elapsed); activeRequest = null; controls.forEach(c => c.disabled = false); $('submit').innerHTML = 'Verify claim <span class="spark">✦</span>'; }
};

window.biomedicalReady = true;
$('submit').disabled = false;
})();
