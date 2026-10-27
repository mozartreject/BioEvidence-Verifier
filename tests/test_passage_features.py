import unittest
import numpy as np
from passage_features import overlap_features,extract

class Tests(unittest.TestCase):
    def test_missing_measurement_and_negation_are_distinguished(self):
        f=overlap_features('Treatment does not reduce mortality by 20 percent','Study','Treatment reduced mortality by 10 percent')
        self.assertEqual(f[3],1);self.assertEqual(f[4],0);self.assertEqual(f[6],1)
    def test_unknown_annotation_cannot_change_features(self):
        pairs=[dict(document_id=str(i),raw=[.1,.8,.1],relevance=.9-i*.2,title='Study',selected={'text':'A treatment study.'}) for i in range(3)]
        record=dict(claim='Treatment helps.',pairs=pairs)
        before=extract(record,[1.,0.,-1.]);record['expected']='SUPPORTED'
        np.testing.assert_array_equal(before,extract(record,[1.,0.,-1.]))

if __name__=='__main__':unittest.main()
