import numpy as np
from numpy.random import default_rng
import unittest

from part1_nn_lib import LinearLayer, MSELossLayer

# learning_rate = 0.0001
# n_samples = 10
# n_in = 2
# n_out = 1
# epochs = 100000

# seed = 60012
# rg = default_rng(seed)

# weights = np.array([4, 2.5])
# bias = np.ones((n_samples)) * 1.5
# x = rg.random((n_samples, n_in))*10.0
# y = np.matmul(x, weights) + bias

# # add noise to y
# # comment these out if you want to work with a perfectly clean dataset
# # noise = rg.standard_normal(y.shape)
# # y = y + noise

# x_train = x
# y_train = y.reshape(-1, 1)
# # x_test = x[80:, :2]
# # y_test = y[80:]

# layer = LinearLayer(n_in, n_out)
# loss = MSELossLayer()

# print(layer._W)
# print(layer._b)

# for x in range(epochs):
#     outputs = layer(x_train)
#     cur_loss = loss.forward(outputs, y_train)

#     if x % 500 == 0:
#         print(f"Epoch: {x}. Loss: {cur_loss}")

#     grad_loss_wrt_outputs = loss.backward()

#     grad_loss_wrt_inputs = layer.backward(grad_loss_wrt_outputs)
#     layer.update_params(learning_rate)

# print(layer._W)
# print(layer._b)

# print(layer(inputs))

class TestNeuralNetworkTraining(unittest.TestCase):
    def setUp(self):
        self.learning_rate = 0.0001
        self.n_samples = 10
        self.n_in = 2
        self.n_out = 1
        self.epochs = 100000

        seed = 60012
        rg = default_rng(seed)

        weights = np.array([4, 2.5])
        bias = np.ones((self.n_samples)) * 1.5
        x = rg.random((self.n_samples, self.n_in)) * 10.0
        y = np.matmul(x, weights) + bias

        self.x_train = x
        self.y_train = y.reshape(-1, 1)

        self.layer = LinearLayer(self.n_in, self.n_out)
        self.loss = MSELossLayer()

    def test_loss_decreases(self):
        initial_loss = self.loss.forward(self.layer(self.x_train), self.y_train)
        final_loss = initial_loss

        for epoch in range(self.epochs):
            outputs = self.layer(self.x_train)
            cur_loss = self.loss.forward(outputs, self.y_train)

            grad_loss_wrt_outputs = self.loss.backward()
            grad_loss_wrt_inputs = self.layer.backward(grad_loss_wrt_outputs)
            self.layer.update_params(self.learning_rate)

            if epoch == self.epochs - 1:
                final_loss = cur_loss

        self.assertLess(final_loss, initial_loss, "Final loss should be less than initial loss")

    def test_learned_parameters(self):
        expected_weights = np.array([4, 2.5])
        expected_bias = 1.5

        for epoch in range(self.epochs):
            outputs = self.layer(self.x_train)
            cur_loss = self.loss.forward(outputs, self.y_train)

            grad_loss_wrt_outputs = self.loss.backward()
            grad_loss_wrt_inputs = self.layer.backward(grad_loss_wrt_outputs)
            self.layer.update_params(self.learning_rate)

        learned_weights = self.layer._W.flatten()
        learned_bias = self.layer._b.flatten()[0]

        np.testing.assert_almost_equal(learned_weights, expected_weights, decimal=1, err_msg="Learned weights do not match expected weights")
        self.assertAlmostEqual(learned_bias, expected_bias, places=1, msg="Learned bias does not match expected bias")

if __name__ == '__main__':
    unittest.main()
