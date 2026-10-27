import json,tempfile,unittest
from pathlib import Path
from core import feature_vector
from predict import predict

class Tests(unittest.TestCase):
    def test_explicit_class_mapping_and_shape_validation(self):
        pairs=[dict(raw=[.2,.5,.3],relevance=.8)]*3
        n=len(feature_vector(pairs))
        artifact=dict(status='experimental_not_deployed',classes=[0,1,2],mean=[0]*n,scale=[1]*n,
                      coefficients=[[0]*n]*3,intercept=[0,2,0],fingerprint='test')
        with tempfile.TemporaryDirectory() as folder:
            p=Path(folder)/'model.json';p.write_text(json.dumps(artifact))
            self.assertEqual(predict(pairs,p)['verdict'],'NEI')
            artifact['scale']=[0]*n;p.write_text(json.dumps(artifact))
            with self.assertRaises(ValueError):predict(pairs,p)
            artifact['classes']=[1,0,2];p.write_text(json.dumps(artifact))
            with self.assertRaises(ValueError):predict(pairs,p)

if __name__=='__main__':unittest.main()
