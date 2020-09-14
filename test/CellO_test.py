import unittest
import numpy as np
import sys

sys.path.append('..')

import CellO

class TestCellO(unittest.TestCase):

    def test__aggregate_expression(self):
        X1 = np.array([[1,2], [1,2]])
        units1 = 'CPM'
        true = (
            12.716901269291665,
            13.410046949854985
        )
        out = CellO._aggregate_expression(X1, units1)
        self.assertAlmostEqual(true[0], out[0], delta=0.000001)
        self.assertAlmostEqual(true[1], out[1], delta=0.000001)

    def test__expression_matrix_subset(self):

        # Test 1
        X = np.array([
            [1,1,1,1], 
            [1,1,1,1], 
            [1,1,1,1]
        ])
        genes = ['a', 'b', 'c']
        gene_to_indices = {
            'a': [0],
            'b': [1,2],
            'c': [3]
        }
        X_correct = np.array([
            [1,2,1],
            [1,2,1],
            [1,2,1]
        ])
        X_res = CellO._expression_matrix_subset(X, genes, gene_to_indices)
        np.testing.assert_array_equal(X_res, X_correct)

        # Test 2
        X = np.array([
            [1,2,3,4],
            [1,2,3,4],
            [1,2,3,4]
        ])
        genes = ['d', 'c', 'b', 'a']
        gene_to_indices = {
            'a': [0],
            'b': [1],
            'c': [2],
            'd': [3]
        }
        X_correct = np.array([
            [4,3,2,1],
            [4,3,2,1],
            [4,3,2,1],
        ])
        X_res = CellO._expression_matrix_subset(X, genes, gene_to_indices)
        np.testing.assert_array_equal(X_res, X_correct)

        # Test 3
        X = np.array([
            [1,2,3,4],
            [1,2,3,4],
            [1,2,3,4]
        ])
        genes = ['b', 'a', 'c']
        gene_to_indices = {
            'a': [0, 3],
            'b': [1],
            'c': [2],
        }
        X_correct = np.array([
            [2,5,3],
            [2,5,3],
            [2,5,3],
        ])
        X_res = CellO._expression_matrix_subset(X, genes, gene_to_indices)
        np.testing.assert_array_equal(X_res, X_correct)

if __name__ == '__main__':
    unittest.main()
