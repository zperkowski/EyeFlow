from unittest import TestCase
from numpy import array, array_equal

from Eye import Eye

test_image = array([
    [0.93, 0.95, 0.97, 0.99],
    [0.73, 0.75, 0.77, 0.79],
    [0.43, 0.45, 0.47, 0.49],
    [0.13, 0.15, 0.17, 0.19]
])

test_batches = [
    array([[0.93, 0.95],
           [0.73, 0.75]]),
    array([[0.95, 0.97],
           [0.75, 0.77]]),
    array([[0.97, 0.99],
           [0.77, 0.79]]),
    array([[0.73, 0.75],
           [0.43, 0.45]]),
    array([[0.75, 0.77],
           [0.45, 0.47]]),
    array([[0.77, 0.79],
           [0.47, 0.49]]),
    array([[0.43, 0.45],
           [0.13, 0.15]]),
    array([[0.45, 0.47],
           [0.15, 0.17]]),
    array([[0.47, 0.49],
           [0.17, 0.19]])
]


class TestEye(TestCase):

    def test__get_batches(self):
        test_eye = Eye(test_image, test_image, test_image, 2)
        batches = test_eye.get_batches_of_raw()
        self.assertTrue(array_equal(batches, test_batches))

    def test_build_image_from_batches(self):
        test_eye = Eye(test_image, test_image, test_image, 2)
        test_batches = test_eye.get_batches_of_raw()
        image = test_eye.build_image_from_batches(test_batches)
        self.assertTrue(array_equal(test_image, image))

    def test__get_batches_too_big(self):
        test_eye = Eye(test_image, test_image, test_image, 10)
        batches = test_eye.get_batches_of_raw()
        self.assertTrue(array_equal(batches, [test_image]))

    def test_build_image_from_batches_too_big(self):
        test_eye = Eye(test_image, test_image, test_image, 10)
        test_batches = test_eye.get_batches_of_raw()
        image = test_eye.build_image_from_batches(test_batches)
        self.assertTrue(array_equal(test_image, image))

    def test_generate_batches_of_raw(self):
        test_eye = Eye(test_image, test_image, test_image, 2)
        self.assertEqual(len(test_batches), len(test_eye.get_batches_of_raw()))
        self.assertEqual(len(test_batches), len(test_eye.get_batches_of_raw()))

    def test_generate_batches_of_manual(self):
        test_eye = Eye(test_image, test_image, test_image, 2)
        self.assertEqual(len(test_batches), len(test_eye.get_batches_of_manual()))
        self.assertEqual(len(test_batches), len(test_eye.get_batches_of_manual()))

    def test_generate_batches_of_manual_too_big(self):
        test_eye = Eye(test_image, test_image, test_image, 10)
        self.assertEqual(1, len(test_eye.get_batches_of_manual()))
