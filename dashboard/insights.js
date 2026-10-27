/* Charts use only response fields; missing values are never replaced by zero. */
function buildInsights(data, evidence) {
 const root = document.createElement('div'); root.className = 'insights';
 const el = (tag, value, cls) => { const n = document.createElement(tag); n.textContent = value; if (cls) n.className = cls; return n; };
 const finite = v => typeof v === 'number' && Number.isFinite(v);
 const unit = v => finite(v) && v >= 0 && v <= 1;
 const section = (title, note) => { const s = el('section', '', 'chart-section'); s.append(el('h3', title), el('p', note, 'chart-note')); root.append(s); return s; };
 const distribution = section('The verdict, in context', 'Model probabilities, not clinical certainty.');
 const keys = ['supported', 'refuted', 'nei']; const names = ['Supported', 'Refuted', 'Insufficient evidence']; const colors = ['#89bfb2', '#b98191', '#6da8ff']; const p = data.phase4?.probabilities;
 if (p && keys.every(k => unit(p[k])) && Math.abs(keys.reduce((sum,k) => sum+p[k],0)-1) <= .01) {
  const layout = el('div', '', 'verdict-chart'); const ring = el('div', '', 'probability-ring'); const sum = keys.reduce((s,k) => s+p[k],0); let cursor = 0;
  ring.style.background = `conic-gradient(${keys.map((k,i) => { const start=cursor; cursor += p[k]/sum*100; return `${colors[i]} ${start}% ${cursor}%`; }).join(',')})`;
  ring.setAttribute('role','img'); ring.setAttribute('aria-label', keys.map((k,i)=>`${names[i]}: ${(p[k]*100).toFixed(1)}%`).join('; '));
  const center = el('div', '', 'ring-center'); center.append(el('strong','3'),el('span','outcomes')); ring.append(center); const legend = el('div','','chart-legend');
  keys.forEach((k,i)=>{const row=el('div','','legend-row');const dot=el('i','');dot.style.background=colors[i];dot.setAttribute('aria-hidden','true');row.append(dot,el('span',names[i]),el('strong',`${(p[k]*100).toFixed(1)}%`));legend.append(row);});layout.append(ring,legend);distribution.append(layout);
 } else distribution.append(el('p','A complete probability distribution was not reported.','chart-empty'));
 const ranking = section('Compare the evidence','Select a measure. Expand any source to read the evidence behind it.');
 const select = document.createElement('select'); select.setAttribute('aria-label','Evidence comparison measure');
 [['relevance_score','Relevance score'],['combined_score','Combined rank score'],['hierarchy_weight','Authority weight']].forEach(([value,title])=>{const o=el('option',title);o.value=value;select.append(o);});ranking.append(select);
 const bars=el('div','','ranking-bars');ranking.append(bars);
 const compare=()=>{bars.replaceChildren();const key=select.value;const valid=evidence.filter(e=>finite(e[key]));const max=Math.max(0,...valid.map(e=>Math.abs(e[key]))); const note=el('p',valid.length?`Bar lengths show score magnitude, relative to the largest reported value. ${valid.length} of ${evidence.length} sources have this measure.`:'This measure was not reported for these sources.','chart-note');bars.append(note);
 [...evidence].sort((a,b)=>finite(a[key])&&finite(b[key])?b[key]-a[key]:finite(a[key])?-1:finite(b[key])?1:0).forEach(e=>{const d=document.createElement('details');d.className='ranking-source';const summary=el('summary','');const heading=el('div','','ranking-label');heading.append(el('span',e.title||e.evidence_id||'Untitled source'),el('strong',finite(e[key])?e[key].toFixed(3):'Not reported'));summary.append(heading);if(finite(e[key])){const track=el('div','','chart-track');const fill=el('div','','chart-fill');fill.style.width=`${max?Math.abs(e[key])/max*100:0}%`;track.append(fill);summary.append(track);}d.append(summary,el('p',e.text||'No excerpt available.','chart-excerpt'),el('p',`Authority tier: ${e.authority_tier||'Not reported'} · ${e.source||'Source not reported'}`,'chart-note'));bars.append(d);});};select.onchange=compare;compare();
 const timing=section('Where the time went','Reported duration per phase. Lengths compare phases; they do not imply sequential execution.');
 const stages=['Represent','Retrieve','Rank','Verify','Explain']; const values=stages.map((_,i)=>data[`phase${i+1}`]?.timing?.latency_ms);const maxTime=Math.max(0,...values.filter(v=>finite(v)&&v>=0));
 stages.forEach((name,i)=>{const row=el('div','','timing-chart-row');row.append(el('span',name));const track=el('div','','chart-track');const v=values[i];if(finite(v)&&v>=0){const fill=el('div','','chart-fill');fill.style.width=`${maxTime?v/maxTime*100:0}%`;fill.style.opacity=String(.5+i*.1);track.append(fill);}row.append(track,el('strong',finite(v)&&v>=0?`${(v/1000).toFixed(2)}s`:'—'));timing.append(row);});
 if(values.some(v=>!finite(v)||v<0))timing.append(el('p','— indicates a duration that was not reported.','chart-note'));
 if(finite(data.total_latency_ms)&&data.total_latency_ms>=0)timing.append(el('p',`Total pipeline time: ${(data.total_latency_ms/1000).toFixed(2)}s. May include overhead or overlapping stages.`,'chart-note'));
 return root;
}
