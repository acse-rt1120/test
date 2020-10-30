import numpy as np
import pytest

from acse_la import gauss, zeromat, matmul



class TestGauss(object):
    """
    Class for testing the Gaussian elimination algorithm Gauss
    and its associated functions.
    """

    @pytest.mark.parametrize('a1, b1, det1e, x1e, a2, b2, det2e, x2e, ce, p1, q1, y1e, p2, q2, y2e', [
        ([[2, 9, 4], [7, 5, 3], [6, 1, 8]],
         [[1, 0, 0], [0, 1, 0], [0, 0, 1]], -360.0,
         [[-0.10277777777777776, 0.18888888888888888, -0.019444444444444438],
          [0.10555555555555554, 0.02222222222222223, -0.061111111111111116],
          [0.0638888888888889, -0.14444444444444446, 0.14722222222222223]],
         [[1, 9.8, -1], [-2, 3, 0], [1, -3, 2]],
         [[1, 0, 0], [0, 1, 0], [0, 0, 1]], 42.2,
         [[0.14218009478672985, -0.3933649289099526, 0.07109004739336493],
          [0.0947867298578199, 0.07109004739336493, 0.04739336492890995],
          [0.07109004739336491, 0.30331753554502366, 0.5355450236966824]],

         [[2, 9, 4], [7, 5, 3], [6, 1, 8]], 3, 3,
         [[0, 0, 0], [0, 0, 0], [0, 0, 0]], 4, 6,
         [[0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0]]
         )
    ])
    def test_gauss(self, a1, b1, det1e, x1e, a2, b2, det2e, x2e, ce, p1, q1, y1e, p2, q2, y2e):
        """ Test the gauss function """
        det1, x1 = gauss(a1, b1)
        det2, x2 = gauss(a2, b2)
        c = matmul(a1, b1)
        y1 = zeromat(p1, q1)
        y2 = zeromat(p2, q2)

        assert np.isclose(det1, det1e)
        assert np.allclose(x1, x1e)
        assert np.isclose(det2, det2e)
        assert np.allclose(x2, x2e)
        assert np.allclose(c, ce)
        assert np.allclose(y1, y1e)
        assert np.allclose(y2, y2e)



