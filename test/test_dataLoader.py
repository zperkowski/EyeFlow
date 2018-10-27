from unittest import TestCase
from DataLoader import DataLoader


class TestDataLoader(TestCase):
    def test_generate_list_of_numbers(self):
        start = 1
        end = 4
        expect = ["01", "02", "03", "04"]
        dl = DataLoader()
        result = dl.generate_list_of_numbers(start, end)
        self.assertEqual(result, expect)

