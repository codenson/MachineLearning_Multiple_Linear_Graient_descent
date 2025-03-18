# test_main.py
import unittest
# from main import compute_cost
from main import addition

class TestMain(unittest.TestCase):
    def test_hello_world(self):
        self.assertEqual(1 + 1, 2)
    # def test_compute_cost(self):
        # self.assertEqual(compute_cost(1, 1, 1, 1), 1)
    def test_addition(self):
        self.assertEqual(addition(1, 1), 2)
        self.assertNotEqual(addition(1, 1), 3)

if __name__ == '__main__':
    unittest.main()