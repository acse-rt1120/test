import numpy as np
import pytest
import doctest

from acse_la import gauss, zeromat, matmul

"""gauss_doc = gauss.__doc__
add_str = '>>>from acse_la import gauss'
gauss_doc_full = add_str + gauss_doc
my_file = open('gauss_doc_full.txt', 'w', encoding='utf-8')
my_file.write(gauss_doc_full)

doctest.testfile("gauss_doc_full.txt")"""

class TestGaussDocstring(object):
    """
    Class for testing the Gaussian elimination algorithm Gauss
    and its associated functions.
    """

    @pytest.mark.parametrize
    def test_gauss(self,gauss):
        return None




#TestGaussDocstring.test_gauss(gauss)
