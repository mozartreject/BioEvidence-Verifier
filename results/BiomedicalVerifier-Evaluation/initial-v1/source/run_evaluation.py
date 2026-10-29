import csv,json,time,uuid,requests
from pathlib import Path
from datetime import datetime,timezone
from collections import Counter
ROOT=Path('/Users/joshua/Desktop/BiomedicalVerifier-Evaluation')
manifest=ROOT/'manifest.json'
items=json.loads(manifest.read_text()) if manifest.exists() else []
SUPPORTED=[
'Malaria is caused by Plasmodium parasites.',
'Malaria is transmitted by infected female Anopheles mosquitoes.',
'Cigarette smoking causes lung cancer.',
'Insulin lowers blood glucose.',
'Type 1 diabetes is an autoimmune disease.',
'Tuberculosis is caused by bacteria.',
'Influenza is caused by viruses.',
'HIV attacks the immune system.',
'High blood pressure increases the risk of stroke.',
'Physical activity reduces the risk of type 2 diabetes.',
'Exposure to ultraviolet radiation increases the risk of skin cancer.',
'The human papillomavirus can cause cervical cancer.',
'Helicobacter pylori infection can cause peptic ulcers.',
'Iron deficiency can cause anemia.',
'Vitamin C deficiency causes scurvy.',
'Vitamin D deficiency can cause rickets in children.',
'Rabies is caused by a virus.',
'Coronary artery disease can cause a heart attack.',
'Obesity increases the risk of type 2 diabetes.',
'Measles is caused by a virus.',
'Diabetes causes high blood glucose.',
'Tetanus is caused by Clostridium tetani.',
'Cholera is caused by Vibrio cholerae.',
'Chickenpox is caused by the varicella-zoster virus.',
'Hepatitis B is caused by a virus.',
'Dehydration occurs when the body loses too much water.',
'Anaphylaxis is a severe allergic reaction.',
'Osteoporosis makes bones more likely to fracture.',
'Asthma causes inflammation of the airways.',
'Kidney stones can cause severe pain.',
'Antibiotics treat bacterial infections.',
'Antibiotics do not treat viruses.',
'Tobacco smoke contains carcinogens.',
'Atherosclerosis involves plaque buildup in arteries.',
'Parkinson disease involves loss of dopamine-producing neurons.',
'Human red blood cells contain hemoglobin.',
'Platelets help blood clot.',
'The pancreas produces insulin.',
'The kidneys filter waste from the blood.',
'The lungs exchange oxygen and carbon dioxide.'
]
REFUTED=[
'Cigarette smoking does not cause lung cancer.',
'Insulin raises blood glucose.',
'Malaria is caused by bacteria.',
'Tuberculosis is caused by a virus.',
'Influenza is caused by bacteria.',
'Antibiotics kill the viruses that cause the common cold.',
'HIV strengthens the immune system.',
'High blood pressure reduces the risk of stroke.',
'Obesity protects against type 2 diabetes.',
'Vitamin C deficiency causes rickets.',
'Vitamin D deficiency causes scurvy.',
'Rabies is caused by bacteria.',
'Measles is caused by bacteria.',
'Cholera is caused by a virus.',
'Tetanus is caused by a virus.',
'Chickenpox is caused by bacteria.',
'Hepatitis B is caused by bacteria.',
'Exposure to ultraviolet radiation cannot cause skin cancer.',
'Iron deficiency cannot cause anemia.',
'The pancreas does not produce insulin.',
'Antibiotics cure influenza by killing influenza viruses.',
'Tobacco smoke contains no carcinogens.',
'Platelets prevent blood from clotting.',
'The kidneys produce insulin.',
'The lungs do not exchange oxygen and carbon dioxide.',
'Human red blood cells do not contain hemoglobin.',
'Osteoporosis makes bones stronger.',
'Coronary artery disease cannot cause a heart attack.',
'Untreated high blood pressure protects the kidneys.',
'Secondhand tobacco smoke cannot cause lung cancer.',
'Malaria is transmitted primarily by infected male Anopheles mosquitoes.',
'Type 1 diabetes is caused by excessive insulin production.',
'Antibiotics are effective against viruses.',
'Insulin resistance means that cells respond more strongly to insulin.',
'Glaucoma never damages the optic nerve.',
'Cataracts make the lens of the eye clearer.',
'Anaphylaxis is always a mild allergic reaction.',
'Helicobacter pylori infection protects against peptic ulcers.',
'Atherosclerosis prevents plaque buildup in arteries.',
'Vaccines cause autism.'
]
NEI=[
'Walking for exactly 17 minutes each morning reduces Alzheimer disease incidence by exactly 23% in adults aged 41 to 43.',
'Eating 12 blueberries daily improves working memory by exactly 18% after nine days.',
'Drinking green tea at 7:15 a.m. prevents migraines better than drinking it at 8:15 a.m.',
'Sleeping on the left side increases antibody responses to influenza vaccination by exactly 14% in adults aged 30 to 35.',
'Taking oral probiotics prevents Parkinson disease in healthy adults who have never used antibiotics.',
'Eating seven almonds at bedtime increases lung capacity by exactly 11% in healthy violinists.',
'Drinking 180 milliliters of water before breakfast improves vaccine antibody responses by exactly 16%.',
'Listening to piano music for 13 minutes lowers fasting glucose by exactly 9% in adults aged 28 to 31.',
'Wearing blue socks improves insulin sensitivity by exactly 12% after six weeks.',
'Sleeping with the bedroom door open reduces migraine frequency by exactly 19% in left-handed adults.',
'Eating one pear on alternate Tuesdays prevents rheumatoid arthritis in healthy adults.',
'Drinking chamomile tea from a red cup increases hemoglobin by exactly 8% after eleven days.',
'Brushing teeth at exactly 6:40 a.m. reduces influenza infection risk by exactly 27%.',
'Taking medicine Zentarovil reduces migraine frequency by exactly 42%.',
'Clapping for three minutes before lunch improves iron absorption by exactly 17%.',
'Eating 23 grams of oats at 9 p.m. prevents glaucoma in healthy adults aged 32 to 36.',
'Sitting facing north improves influenza vaccine effectiveness by exactly 21%.',
'Walking barefoot on indoor tiles for eight minutes reduces asthma attacks by exactly 13%.',
'Changing pillowcases every 36 hours prevents type 1 diabetes in adults.',
'Drinking warm water from a glass cup reverses insulin resistance within exactly five days.'
]
# Run a balanced first set, then reserves only for classes still below quota.
queue=[]
for i in range(20):
 for label,claims in [('SUPPORTED',SUPPORTED),('REFUTED',REFUTED),('NEI',NEI)]:queue.append((label,claims[i]))
for i in range(20,40):
 for label,claims in [('SUPPORTED',SUPPORTED),('REFUTED',REFUTED)]:queue.append((label,claims[i]))

def save():
 temp=manifest.with_suffix('.tmp');temp.write_text(json.dumps(items,indent=2));temp.replace(manifest)
 fields=['sequence','request_id','claim','expected_candidate','actual_verdict','selected','explanation_status','quarantine_reason','http_status','error','elapsed_seconds','raw_file','screenshot']
 with (ROOT/'results.csv').open('w',newline='') as f:
  w=csv.DictWriter(f,fieldnames=fields,extrasaction='ignore');w.writeheader();w.writerows(items)
 (ROOT/'README.txt').write_text('Actual system evaluation. Claims are grouped by returned verdict, not expected label. Scores are uncalibrated AI estimates.\nQuarantined means the explanation was withheld; the verdict can still be recorded. Failed reviews are not NEI.\nEvery run has a raw response and request ID. Extra runs and disagreements remain in results.csv and manifest.json.\nScreenshots load genuine saved responses through the dashboard archive view. No verdicts, percentages or evidence are substituted.\nCounts: '+str(dict(Counter(i['actual_verdict'] for i in items if i.get('selected'))))+'\n')
seen={i['claim'] for i in items}
for expected,claim in queue:
 if claim in seen:continue
 counts=Counter(i['actual_verdict'] for i in items if i.get('selected'))
 quarantine_count=sum(i.get('explanation_status')=='QUARANTINED' for i in items)
 if all(counts[k]>=20 for k in ('SUPPORTED','REFUTED','NEI')) and quarantine_count>=3:break
 if expected=='NEI' and counts['NEI']>=20:continue
 if len(items)>=60 and counts[expected]>=20:continue
 rid=str(uuid.uuid4());start=time.time()
 print('START',len(items)+1,expected,claim,flush=True)
 item=dict(sequence=len(items)+1,request_id=rid,claim=claim,expected_candidate=expected,actual_verdict='FAILED_REVIEW',selected=False,started_at_utc=datetime.now(timezone.utc).isoformat(),screenshot='')
 try:
  response=requests.post('http://127.0.0.1:8003/api/v1/inferences',data={'claim':claim,'request_id':rid},timeout=(5,600))
  item['http_status']=response.status_code
  try:data=response.json()
  except ValueError:data={'error':response.text[:4000]}
  file=ROOT/'raw'/(rid+'.json');file.write_text(json.dumps(data,indent=2));item['raw_file']=str(file.relative_to(ROOT))
  if response.ok and data.get('final_verdict') in ('SUPPORTED','REFUTED','NEI'):
   verdict=data['final_verdict'];item['actual_verdict']=verdict;item['selected']=counts[verdict]<20
   item['explanation_status']=data.get('phase5',{}).get('explanation_status')
   item['quarantine_reason']=data.get('phase5',{}).get('quarantine_reason')
   item['model']=data.get('phase4',{}).get('model',{}).get('model_name')
  else:item['error']=str(data.get('detail',data))[:4000]
 except Exception as exc:item['error']=str(exc)
 item['elapsed_seconds']=round(time.time()-start,2);items.append(item);seen.add(claim);save()
 print('DONE',item['sequence'],item['actual_verdict'],item.get('explanation_status'),dict(Counter(i['actual_verdict'] for i in items if i.get('selected'))),flush=True)
save();print('BATCH_FINISHED',flush=True)
