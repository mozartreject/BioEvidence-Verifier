import unittest
import numpy as np
from core import SparseBM25, linked_groups, feature_vector, baselines

class Tests(unittest.TestCase):
    def test_bm25_matches_hand_calculation(self):
        texts=['alpha alpha beta','beta gamma','delta'];r=SparseBM25(texts)
        expected=2*2.5/(2+1.5*(.25+.75*3/2))*np.log1p((3-1+.5)/(1+.5))
        result=r.search('alpha');self.assertEqual(result[0][0],0);self.assertAlmostEqual(result[0][1],expected)
        self.assertEqual(r.search('unseen'),[])
    def test_transitive_document_groups(self):
        c=[dict(id=1,cited_doc_ids=[1]),dict(id=2,cited_doc_ids=[1,2]),dict(id=3,evidence={'2':[]}),dict(id=4,cited_doc_ids=[4])]
        g=linked_groups(c);self.assertEqual(g[0],g[2]);self.assertNotEqual(g[0],g[3])
    def test_features_ignore_annotation_and_identity(self):
        p=[dict(raw=[.1,.2,.7],relevance=r) for r in [.8,.4,.2]]
        a=feature_vector(p)
        for item in p:item.update(gold_label='REFUTED',document_id='secret')
        np.testing.assert_array_equal(a,feature_vector(p[::-1]))
    def test_zero_relevance_is_nei(self):
        p=[dict(raw=[1,0,0],relevance=0) for _ in range(3)]
        self.assertEqual(baselines(p),dict(mean=1,top_document=1))
    def test_reject_invalid_probabilities(self):
        with self.assertRaises(ValueError):feature_vector([dict(raw=[.2,.1,.1],relevance=.9)]*3)

if __name__=='__main__':unittest.main()
