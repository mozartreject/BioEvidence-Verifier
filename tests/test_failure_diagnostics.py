import asyncio,unittest
from failure_diagnostics import failure_context,build_diagnostics
from relation_review import RelationReviewError
class FailureDiagnosticsTests(unittest.TestCase):
 def test_model_result_is_explicitly_rejected(self):
  result=dict(verdict='REFUTED',scores={'REFUTED':.8},corpus_documents=35183,evidence=[dict(document_id='pubmed:123',title='Paper',source='pubmed',selected={'text':'Selected passage'},raw=[.01,.98,.01],relevance=.7)])
  d=build_diagnostics('Claim',result,ValueError('Evidence validation failed'))
  self.assertIsNone(d['accepted_verdict']);self.assertEqual(d['classification_status'],'REJECTED');self.assertIn('neutral',d['explanation']);self.assertEqual(d['evidence'][0]['source_url'],'https://pubmed.ncbi.nlm.nih.gov/123/')
 def test_reviewer_disagreement_preserved(self):
  review={'verdict':'NEI','reason':'Different outcome','citations':[]}
  d=build_diagnostics('Claim',{'evidence':[]},RelationReviewError('Disagreement',review))
  self.assertEqual(d['relation_review'],review)
 def test_thread_context_is_request_local(self):
  async def check():
   async def request(value):
    token=failure_context.set({})
    try:
     await asyncio.to_thread(lambda:failure_context.get().update(value=value))
     await asyncio.sleep(0)
     self.assertEqual(failure_context.get()['value'],value)
    finally:failure_context.reset(token)
   await asyncio.gather(request('one'),request('two'))
  asyncio.run(check());self.assertIsNone(failure_context.get())
if __name__=='__main__':unittest.main()
