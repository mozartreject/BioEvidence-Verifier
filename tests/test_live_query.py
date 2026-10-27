import unittest
from live_query import compact_terms


class CompactQueryTests(unittest.TestCase):
    def test_broadens_long_assertion_without_changing_entity_endpoints(self):
        self.assertEqual(compact_terms('antibiotics against viruses common cold'), 'antibiotics common cold')
        self.assertEqual(compact_terms('smoking increases lung cancer risk'), 'smoking cancer risk')

    def test_short_queries_do_not_add_a_second_request(self):
        self.assertEqual(compact_terms('vaccines autism'), '')
        self.assertEqual(compact_terms(''), '')

    def test_removes_generic_population_and_absolute_words_only_in_search(self):
        self.assertEqual(compact_terms('people heart disease never exercise'), 'heart disease exercise')


if __name__ == '__main__':
    unittest.main()
