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

if __name__ == '__main__':
    unittest.main()
