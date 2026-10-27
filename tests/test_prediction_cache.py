import unittest
from functools import lru_cache
from candidate import Candidate
class CacheTests(unittest.TestCase):
 def engine(self,live=False):
  c=Candidate.__new__(Candidate);c.live_pubmed=live;c.calls=0
  def work(claim):
   c.calls+=1;return {'claim':claim,'scores':[1,2,3]}
  c._predict=work;c._cached_predict=lru_cache(maxsize=16)(work);return c
 def test_cached_results_are_isolated(self):
  c=self.engine();r=c.predict('one');r['scores'][0]=999
  self.assertEqual(c.predict('one')['scores'][0],1);self.assertEqual(c.calls,1)
 def test_explicit_cache_clear(self):
  c=self.engine();c.predict("one");c.clear_cache();c.predict("one")
  self.assertEqual(c.calls,2)
 def test_separate_instances_do_not_share_cache(self):
  first=self.engine();second=self.engine();first.predict("one")
  self.assertEqual(second._cached_predict.cache_info().currsize,0)
 def test_live_bypasses_cache(self):
  c=self.engine(True);c.predict('one');c.predict('one');self.assertEqual(c.calls,2)
 def test_bounded_and_exact_claims(self):
  c=self.engine()
  for i in range(20):c.predict(str(i))
  self.assertEqual(c._cached_predict.cache_info().currsize,16)
  c.predict('0');self.assertEqual(c.calls,21)
  c.predict('Not supported');c.predict('supported');self.assertEqual(c.calls,23)
if __name__=='__main__':unittest.main()
