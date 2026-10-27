import itertools
import unittest
from nli_contract import canonical_order, CANONICAL_NLI

class LabelContractTests(unittest.TestCase):
    def test_every_model_label_permutation(self):
        for labels in itertools.permutations(CANONICAL_NLI):
            mapping=dict(enumerate(labels));order=canonical_order(mapping)
            for semantic in CANONICAL_NLI:
                raw=[float(label==semantic) for label in labels]
                canonical=[raw[index] for index in order]
                self.assertEqual(canonical.index(1.0),CANONICAL_NLI.index(semantic))
    def test_reported_config(self):
        self.assertEqual(canonical_order({'0':'contradiction','1':'entailment','2':'neutral'}),[0,2,1])
    def test_ambiguous_configs_fail(self):
        for mapping in [{0:'LABEL_0',1:'LABEL_1',2:'LABEL_2'},
                        {0:'contradiction',1:'neutral',2:'neutral'},
                        {1:'contradiction',2:'neutral',3:'entailment'}]:
            with self.assertRaises(ValueError):canonical_order(mapping)
    def test_wrong_output_size_fails(self):
        with self.assertRaises(ValueError):canonical_order(dict(enumerate(CANONICAL_NLI)),4)

if __name__=='__main__':unittest.main()
