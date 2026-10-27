import unittest
from relation_review import validate_review,require_agreement
class ReviewTests(unittest.TestCase):
 def setUp(self):self.docs=[dict(evidence_id='a',text='Reducing exposure decreased the measured outcome.')]
 def test_exact_citation_required(self):
  for quote in ['exposure prevents the outcome','',None]:
   with self.assertRaises(ValueError):validate_review(dict(verdict='SUPPORTED',reason='Because',citations=[dict(evidence_id='a',quote=quote)]),self.docs)
 def test_unknown_evidence_rejected(self):
  with self.assertRaises(ValueError):validate_review(dict(verdict='REFUTED',reason='Opposite action',citations=[dict(evidence_id='b',quote=self.docs[0]['text'])]),self.docs)
 def test_decisive_review_requires_citation(self):
  with self.assertRaises(ValueError):validate_review(dict(verdict='SUPPORTED',reason='Reason',citations=[]),self.docs)
 def test_valid_quote(self):
  r=validate_review(dict(verdict='REFUTED',reason='Opposite direction',citations=[dict(evidence_id='a',quote='Reducing exposure')]),self.docs)
  self.assertEqual(r['verdict'],'REFUTED')
 def test_whitespace_only_citation_repair(self):
  docs=[dict(evidence_id='a',text='Review title.\nBACKGROUND: Antibiotics have no role.')]
  r=validate_review(dict(verdict='REFUTED',reason='Opposite direction',citations=[dict(evidence_id='a',quote='Review title. BACKGROUND: Antibiotics have no role.')]),docs)
  self.assertEqual(r['citations'][0]['quote'],docs[0]['text'])
 def test_changed_words_still_rejected(self):
  docs=[dict(evidence_id='a',text='Review title.\nBACKGROUND: Antibiotics have no role.')]
  with self.assertRaisesRegex(ValueError,'Quote does not match'):
   validate_review(dict(verdict='REFUTED',reason='Opposite direction',citations=[dict(evidence_id='a',quote='Review title. BACKGROUND: Antibiotics are effective.')]),docs)
 def test_disagreement_vetoes_without_overriding(self):
  result=dict(verdict='SUPPORTED',evidence=[])
  with self.assertRaisesRegex(ValueError,'disagrees'):require_agreement('claim',result,lambda *_:dict(verdict='REFUTED',reason='Different intervention'))
  self.assertEqual(result['verdict'],'SUPPORTED')
 def test_failure_closed(self):
  def fail(*_):raise TimeoutError()
  with self.assertRaisesRegex(ValueError,'could not be completed'):require_agreement('claim',dict(verdict='SUPPORTED',evidence=[]),fail)
 def test_nei_does_not_call_reviewer(self):
  self.assertIsNone(require_agreement('claim',dict(verdict='NEI'),lambda *_:self.fail('Must not call')))
if __name__=='__main__':unittest.main()
