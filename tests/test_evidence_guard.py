import unittest
from evidence_guard import validate_decisive_evidence
class EvidenceGuardTests(unittest.TestCase):
 def test_decisive_all_neutral_rejected(self):
  for label in ['SUPPORTED','REFUTED']:
   with self.assertRaisesRegex(ValueError,'all retrieved passages'):
    validate_decisive_evidence(dict(verdict=label,evidence=[dict(raw=[.001,.998,.001])]*3))
 def test_opposite_direction_rejected(self):
  for label,raw in [("SUPPORTED",[.99,.005,.005]),("REFUTED",[.005,.005,.99])]:
   with self.assertRaisesRegex(ValueError,"no retrieved passage"):
    validate_decisive_evidence(dict(verdict=label,evidence=[dict(raw=raw)]))
 def test_bad_sum_rejected(self):
  with self.assertRaises(ValueError):validate_decisive_evidence(dict(verdict="SUPPORTED",evidence=[dict(raw=[.2,.2,.2])]))
 def test_nei_unchanged(self):validate_decisive_evidence(dict(verdict='NEI',evidence=[dict(raw=[0,1,0])]))
 def test_directional_evidence_not_blocked(self):
  for raw,label in [([0,0,1],'SUPPORTED'),([1,0,0],'REFUTED')]:
   validate_decisive_evidence(dict(verdict=label,evidence=[dict(raw=raw)]))
 def test_missing_evidence_rejected(self):
  with self.assertRaises(ValueError):validate_decisive_evidence(dict(verdict='SUPPORTED',evidence=[]))
 def test_nonfinite_rejected(self):
  with self.assertRaises(ValueError):validate_decisive_evidence(dict(verdict='SUPPORTED',evidence=[dict(raw=[float('nan'),0,1])]))
if __name__=='__main__':unittest.main()
