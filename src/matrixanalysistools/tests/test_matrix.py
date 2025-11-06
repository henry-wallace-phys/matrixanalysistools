import numpy as np
from matrixanalysistools.matrix_handler.root_matrix import RootMatrix, MatrixFileHandler

# Consts.
SPACE_DIM=4
TEST_MATRIX = np.identity(SPACE_DIM)
TEST_EIG = np.ones(SPACE_DIM)

TRIANGULAR_MATRIX_FLAT = np.arange(10)
TRIANGULAR_MATRIX = np.array( [ np.array([0., 1., 2., 3.,]),
                                np.array([1., 4., 5., 6.,]),
                                np.array([2., 5., 7., 8.]),
                                np.array([3., 6., 8., 9.])
                              ]
                            )

def test_root_matrix():
    matrix = RootMatrix(TEST_MATRIX, "my_matrix")
    assert matrix.name == "my_matrix"
    assert matrix.norm == np.sqrt(SPACE_DIM)
    assert matrix.trace == SPACE_DIM

    # Multiplication tests
    assert matrix*5 == RootMatrix(5*TEST_MATRIX)
    assert matrix*RootMatrix(TRIANGULAR_MATRIX) == RootMatrix(TRIANGULAR_MATRIX)

    np.testing.assert_array_equal(matrix.eigenvalues, TEST_EIG)

    # Symmetry test
    np.testing.assert_array_equal(RootMatrix.tmatrix_dsym_to_numpy(TRIANGULAR_MATRIX_FLAT), TRIANGULAR_MATRIX)

