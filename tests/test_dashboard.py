import unittest
from types import SimpleNamespace
from dashboard_backend import Shared,Retrieval,Verification,create_app

class Tests(unittest.TestCase):
    def setUp(self):
        self.shared=Shared()
        evidence=[dict(document_id='scifact:'+str(i),source='scifact',title='Paper',selected={'text':'Evidence text.'},raw=[.1,.2,.7],bm25_score=2.,reranker_score=1.,relevance=.7) for i in range(3)]
        self.shared.engine=SimpleNamespace(docs=[1,2,3],predict=lambda c:dict(claim=c,verdict='NEI',scores={'REFUTED':.1,'NEI':.6,'SUPPORTED':.3},evidence=evidence))
    def test_contract_order_and_locked_scores(self):
        retrieval=Retrieval(self.shared);retrieval.load();r=retrieval.run('Test claim','id')
        v=Verification(self.shared).run('Test claim',list(reversed(r.evidence)))
        self.assertEqual(v.evidence_used_ids,[e.evidence_id for e in reversed(r.evidence)])
        self.assertEqual(v.verdict.value,'NEI');self.assertEqual(v.verifier_score,.6);self.assertTrue(v.verdict_locked)
    def test_changed_claim_or_evidence_is_rejected(self):
        r=Retrieval(self.shared).run('Test claim','id');v=Verification(self.shared)
        with self.assertRaises(ValueError):v.run('Other claim',r.evidence)
        with self.assertRaises(ValueError):v.run('Test claim',r.evidence[:2])
    def test_app_routes_exist_without_loading_models(self):
        app=create_app();paths=[r.path for r in app.routes]
        self.assertEqual(paths.count('/api/v1/system/health'),1)
        self.assertIn('/api/v1/inferences',paths)

if __name__=='__main__':unittest.main()
