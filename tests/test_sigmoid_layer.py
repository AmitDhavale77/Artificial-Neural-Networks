# TODO: DO NOT SUBMIT THIS FILE

import unittest
import numpy as np
from part1_nn_lib import SigmoidLayer


class TestSigmoidLayer(unittest.TestCase):

    def setUp(self):
        self.layer = SigmoidLayer()

    def test_forward(self):
        x = np.array([[0, 2], [-1, -3]])
        expected_output = 1 / (1 + np.exp(-x))
        output = self.layer.forward(x)
        np.testing.assert_array_almost_equal(output, expected_output, decimal=6)

    def test_backward(self):
        x = np.array([[0, 2], [-1, -3]])
        grad_z = np.array([[1, 1], [1, 1]])
        sig_x = self.layer.forward(x)
        grad = self.layer.backward(grad_z)
        expected_grad = grad_z * (sig_x * (np.ones(x.shape) - sig_x))

        assert grad.shape == x.shape, f"Expected shape {x.shape} but got {grad.shape}"
        np.testing.assert_array_almost_equal(grad, expected_grad, decimal=6)


if __name__ == "__main__":
    unittest.main()
