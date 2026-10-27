/* Exercise initialization and actual async submit code without sending medical text externally. */
const fs = require('fs'), vm = require('vm'), assert = require('assert');
const path = require('path'), root = __dirname;
function element() {
 return {children:[],handlers:{},value:'',files:[],hidden:false,disabled:false,style:{},dataset:{},innerHTML:'',
 addEventListener(name,fn){(this.handlers[name] ||= []).push(fn)},querySelectorAll(){return []},
 append(...parts){this.children.push(...parts)},replaceChildren(...parts){this.children=parts},
 setAttribute(){},removeAttribute(){},focus(){}};
}
function setup(response) {
 const nodes={};const html=fs.readFileSync(path.join(root,'index.html'),'utf8');
 for(const m of html.matchAll(/id="([^"]+)"/g)) nodes[m[1]]=element();
 nodes.submit.disabled=true;
 let requests=[];
 const context={document:{getElementById:id=>nodes[id] || (nodes[id]=element()),querySelectorAll:()=>[],addEventListener(){},createElement:()=>element()},
 location:{hostname:'127.0.0.1',origin:'http://127.0.0.1:3001'},localStorage:{getItem:()=>null},
 addEventListener(){},console,FormData,AbortController,crypto:require('crypto').webcrypto,
 setTimeout:()=>1,clearTimeout(){},setInterval:()=>1,clearInterval(){},
 fetch:async(url,opts)=>{requests.push({url,opts});return response},buildInsights:()=>element()};
 context.window=context;vm.createContext(context);
 vm.runInContext(fs.readFileSync(path.join(root,'bootstrap.js'),'utf8'),context);
 return {context,nodes,requests};
}
(async()=>{
 let state=setup();state.nodes.claim.value='Keep this claim';let prevented=false;
 state.nodes['claim-form'].handlers.submit[0]({preventDefault(){prevented=true}});
 assert(prevented);assert(state.nodes.submit.disabled);assert.equal(state.nodes.claim.value,'Keep this claim');
 for(const [status,payload] of [[422,{detail:{error:'EVIDENCE_VALIDATION_FAILED',message:'Evidence does not justify a verdict',diagnostics:{proposed_verdict:'REFUTED',proposed_scores:{REFUTED:.8,NEI:.15,SUPPORTED:.05},explanation:'All passages neutral',scientific_meaning:'A pipeline failure',retrieval:{selected_documents:1,corpus_documents:35183,live_search:false},evidence:[{evidence_id:'pubmed:123',title:'Test paper',text:'Evidence passage',source:'pubmed',source_url:'https://pubmed.ncbi.nlm.nih.gov/123/',probabilities:{refuted:.01,nei:.98,supported:.01},pairwise_label:'NEI',relevance:.7}],limitations:['Diagnostic only']}}}],[503,{detail:'Backend unavailable'}],[200,{request:{claim:'A claim that must stay in the input.'},phase4:{probabilities:{refuted:.1,nei:.8,supported:.1}},final_verdict:'NEI',final_verifier_score:.8,evidence:[]}]]) {
  state=setup({ok:status===200,status,json:async()=>payload});
  vm.runInContext(fs.readFileSync(path.join(root,'app.js'),'utf8'),state.context);
  assert(state.context.biomedicalReady);assert(!state.nodes.submit.disabled);
  state.nodes.claim.value='A claim that must stay in the input.';prevented=false;
  await state.nodes['claim-form'].onsubmit({preventDefault(){prevented=true}});
  assert(prevented);assert.equal(state.requests.length,1);assert.equal(state.requests[0].opts.method,'POST');
  assert.equal(state.requests[0].opts.body.get('claim'),state.nodes.claim.value);
  assert(state.requests[0].opts.body.get('request_id'));assert(!state.nodes.submit.disabled);
  assert.equal(state.nodes.claim.value,'A claim that must stay in the input.');
  if(status===422) {
   assert.equal(state.nodes['result-tag'].textContent,'Verdict withheld');
   const contents = n => [n.textContent || '', ...(n.children || []).map(contents)].join(' ');
   const rendered = contents(state.nodes['result-body']);
   for (const text of ['15.0%', '80.0%', '5.0%', 'Proposed Refuted score', 'Request time', 'Sources', 'No class was accepted']) assert(rendered.includes(text), text);
  }
  else if(status===503) assert.equal(state.nodes.error.textContent,'Backend unavailable');
  else assert.equal(state.nodes['result-tag'].textContent,'Result');
 }
 console.log('PASS: failed initialization blocks navigation; 200/422/503 submissions POST once, preserve the claim, and re-enable Verify.');
})().catch(e=>{console.error(e);process.exitCode=1});
