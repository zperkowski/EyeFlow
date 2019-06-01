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
    array([[[0.93], [0.95]],
           [[0.73], [0.75]]]),
    array([[[0.97], [0.99]],
           [[0.77], [0.79]]]),
    array([[[0.43], [0.45]],
           [[0.13], [0.15]]]),
    array([[[0.47], [0.49]],
           [[0.17], [0.19]]])
]

test_image_odd = array([
    [0.93, 0.95, 0.97, 0.99, 0.3],
    [0.73, 0.75, 0.77, 0.79, 0.32],
    [0.43, 0.45, 0.47, 0.49, 0.33],
    [0.13, 0.15, 0.17, 0.19, 0.35],
    [0.03, 0.05, 0.07, 0.09, 0.25]
])

test_batches_odd = [
    array([[[0.93], [0.95]],
           [[0.73], [0.75]]]),
    array([[[0.97], [0.99]],
           [[0.77], [0.79]]]),
    array([[[0.43], [0.45]],
           [[0.13], [0.15]]]),
    array([[[0.47], [0.49]],
           [[0.17], [0.19]]])
]

test_image_odd_output = array([
    [0.93, 0.95, 0.97, 0.99, 0.],
    [0.73, 0.75, 0.77, 0.79, 0.],
    [0.43, 0.45, 0.47, 0.49, 0.],
    [0.13, 0.15, 0.17, 0.19, 0.],
    [0., 0., 0., 0., 0.]
])

class TestEye(TestCase):

    def test_get_batches(self):
        test_eye = Eye(test_image, test_image, test_image, 2, 2)
        batches = test_eye.get_batches_of_raw()
        self.assertTrue(array_equal(batches, test_batches))

    def test_get_batches_2(self):
        test_eye = Eye(test_image_odd, test_image_odd, test_image_odd, 2, 2)
        batches = test_eye.get_batches_of_raw()
        self.assertTrue(array_equal(batches, test_batches_odd))

    def test_build_image_from_batches(self):
        test_eye = Eye(test_image, test_image, test_image, 2, 2)
        test_batches = test_eye.get_batches_of_raw()
        image = test_eye.build_image_from_batches(test_batches)
        self.assertTrue(array_equal(test_image, image))

    def test_build_image_from_batches_odd(self):
        test_eye = Eye(test_image_odd, test_image_odd, test_image_odd, 2, 2)
        test_batches = test_eye.get_batches_of_raw()
        image = test_eye.build_image_from_batches(test_batches)
        self.assertTrue(array_equal(test_image_odd_output, image))

    def test_get_batches_too_big(self):
        test_eye = Eye(test_image, test_image, test_image, 10, 10)
        batches = test_eye.get_batches_of_raw()
        batch = batches[0].reshape(test_image.shape)
        self.assertEqual(1, len(batches))
        self.assertTrue(array_equal(batch, test_image))

    def test_build_image_from_batches_too_big(self):
        test_eye = Eye(test_image, test_image, test_image, 10, 10)
        test_batches = test_eye.get_batches_of_raw()
        image = test_eye.build_image_from_batches(test_batches)
        self.assertTrue(array_equal(test_image, image))

    def test_generate_batches_of_raw(self):
        test_eye = Eye(test_image, test_image, test_image, 2, 2)
        self.assertEqual(len(test_batches), len(test_eye.get_batches_of_raw()))
        self.assertEqual(len(test_batches), len(test_eye.get_batches_of_raw()))

    def test_generate_batches_of_manual(self):
        test_eye = Eye(test_image, test_image, test_image, 2, 2)
        self.assertEqual(len(test_batches), len(test_eye.get_batches_of_manual()))
        self.assertEqual(len(test_batches), len(test_eye.get_batches_of_manual()))

    def test_generate_batches_of_manual_too_big(self):
        test_eye = Eye(test_image, test_image, test_image, 10, 10)
        self.assertEqual(1, len(test_eye.get_batches_of_manual()))
