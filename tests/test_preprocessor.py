# TODO: DO NOT SUBMIT THIS TEST FILE

import unittest
import numpy as np
from part1_nn_lib import Preprocessor


class TestPreprocessor(unittest.TestCase):

    def setUp(self):
        # Create a sample dataset
        self.data = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        self.preprocessor = Preprocessor(self.data)

    def test_mean_and_std(self):
        # Test if mean and std are calculated correctly
        np.testing.assert_array_almost_equal(
            self.preprocessor.mean, np.array([4, 5, 6])
        )
        np.testing.assert_array_almost_equal(
            self.preprocessor.std, np.array([2.44948974, 2.44948974, 2.44948974])
        )

    def test_apply(self):
        # Test if apply method normalizes the data correctly
        normalized_data = self.preprocessor.apply(self.data)
        expected_normalized_data = (
            self.data - self.preprocessor.mean
        ) / self.preprocessor.std
        np.testing.assert_array_almost_equal(normalized_data, expected_normalized_data)

    def test_revert(self):
        # Test if revert method denormalizes the data correctly
        normalized_data = self.preprocessor.apply(self.data)
        reverted_data = self.preprocessor.revert(normalized_data)
        np.testing.assert_array_almost_equal(reverted_data, self.data)

    def test_identical_values(self):
        # Test that an errror is raised if all values for a feature are identical
        identical_data = np.array([[5, 1, 2], [5, 4, 5], [5, 6, 7]])
        with self.assertRaises(AssertionError):
            preprocessor = Preprocessor(identical_data)


if __name__ == "__main__":
    unittest.main()
