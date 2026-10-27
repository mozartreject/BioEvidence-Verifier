import unittest
from claim_input import retrieval_claim


class ClaimInputTests(unittest.TestCase):
    def test_clipped_population_opening(self):
        self.assertEqual(retrieval_claim('n adults with coronary heart disease'),
                         'In adults with coronary heart disease')

    def test_preserves_other_claims_and_direction(self):
        self.assertEqual(retrieval_claim('Smoking reduces lung cancer risk.'),
                         'Smoking reduces lung cancer risk.')
        self.assertEqual(retrieval_claim('n = 42 patients were enrolled.'),
                         'n = 42 patients were enrolled.')

    def test_whitespace_cleanup(self):
        self.assertEqual(retrieval_claim('In\tadults\n with  heart disease'),
                         'In adults with heart disease')
