import unittest
from retrieval_rescue import select_result


def result(verdict, direction=0):
    raw=[0.05,0.05,0.05]
    raw[direction]=0.9
    return {'verdict':verdict,'evidence':[{'raw':raw}], 'scores':{'REFUTED':0.3,'NEI':0.3,'SUPPORTED':0.4}}


class RescueTests(unittest.TestCase):
    def test_accepts_local_without_network(self):
        local=result('REFUTED')
        selected,review,trace=select_result('claim',local,lambda _:self.fail('unnecessary live search'),
            reviewer=lambda *_:{'verdict':'REFUTED'})
        self.assertIs(selected,local)
        self.assertEqual(trace['selected'],'local')
        self.assertEqual(review['verdict'],'REFUTED')

    def test_retries_rejected_local_and_accepts_reviewed_live(self):
        local=result('SUPPORTED',2);live=result('REFUTED')
        def reviewer(_,candidate):
            if candidate is local:raise ValueError('review disagrees')
            return {'verdict':'REFUTED'}
        selected,review,trace=select_result('claim',local,lambda _:live,reviewer)
        self.assertIs(selected,live)
        self.assertEqual(review['verdict'],'REFUTED')
        self.assertEqual(trace['selected'],'live')

    def test_neutral_live_cannot_override_rejected_local(self):
        local=result('SUPPORTED',2);live=result('REFUTED',1)
        selected,review,trace=select_result('claim',local,lambda _:live,
            reviewer=lambda *_: (_ for _ in ()).throw(ValueError('reject')))
        self.assertIs(selected,local)
        self.assertIsNone(review)
        self.assertEqual(trace['selected'],'local')

    def test_live_failure_retains_local(self):
        local=result('NEI',1)
        selected,review,trace=select_result('claim',local,lambda _: (_ for _ in ()).throw(TimeoutError()),
            reviewer=lambda *_: self.fail('NEI should not be reviewed'))
        self.assertIs(selected,local)
        self.assertIsNone(review)
        self.assertIn('TimeoutError',trace['reason'])

if __name__=='__main__': unittest.main()
