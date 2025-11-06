'''
Objects to handle groups of root matrices
'''

import logging
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from tqdm.rich import tqdm
import uproot
from scipy.linalg import eigvalsh

from matrixanalysistools.exceptions import NonSquareMatrixError, EigenDecompError, ObjectNotFoundError

# ------------------------
class RootMatrix:
    '''
    Wrapper around ROOT matrix to be read from ROOT-file. Assumes matrix is square
    '''
# ------------------------
    def __init__(self, matrix: NDArray, matrix_name: str=""):
        '''
        Constructor

        file: Open uproot filr
        '''
        self._matrix = matrix
        self._name = matrix_name

        if self._matrix.shape[0]!=self._matrix.shape[1]:
            raise NonSquareMatrixError(f"Matrix {matrix_name} is not square ({self._matrix.shape[0]}x{self._matrix.shape[1]})")

        self._matrix_dim = self._matrix.shape[0]
        self._eigenvalues: Optional[NDArray] = None
        self._cholesky_decomp: Optional[RootMatrix] = None
        self._suboptimality: Optional[float] = None

        self._matrix_norm: float = np.linalg.norm(self._matrix)
        
        self._trace = np.trace(self._matrix)

    @classmethod
    def from_root_file(cls, file: uproot.ReadOnlyFile | Path, matrix_name: str):
        if isinstance(file, Path):
            if not Path.exists:
                raise FileNotFoundError(f"Could not find file {file}")

            file = uproot.open(file)

        # WE ASSUME everything is a symmetric TH1D
        matrix_model = file.get(matrix_name, None)

        if matrix_model is None:
            raise ObjectNotFoundError(f"Couldn't find {matrix_name} in {file.file_path}")

        # Output root matrix is a 1D array
        flat_matrix: NDArray = matrix_model.members['fElements']

        matrix_dim = np.sqrt(len(flat_matrix))

        if int(matrix_dim)!=matrix_dim:
            try:
                matrix = cls.tmatrix_dsym_to_numpy(flat_matrix)
            except NonSquareMatrixError as e:
                raise NonSquareMatrixError(f"Matrix {matrix_name} has dim {matrix_dim} [len = {int(matrix_dim**2)}] which is not square!") from e

        else:
            # Now we have our matrix
            matrix = flat_matrix.reshape((matrix_dim, matrix_dim))
        return cls(matrix, matrix_name)

    @classmethod
    def tmatrix_dsym_to_numpy(cls, flat_arr: np.ndarray):
        ''' Converts ROOT TMatrixDSym into usable ROOT object. We're a bit inefficient in that we fill the 
            full square but can fairly comfortably assume this won't make the memory usage too awful
        '''
        # Upper triangular so need to convert to a matrix
        n_dim = int((np.sqrt(8 * len(flat_arr) + 1) - 1) / 2)

        if n_dim!=int(n_dim):
            raise NonSquareMatrixError(f"Input matrix is not upper triangular ({n_dim!=int(n_dim)})")


        full_matrix = np.zeros((n_dim, n_dim))

        iu = np.triu_indices(n_dim)
        full_matrix[iu] = flat_arr

        # Mirror
        return full_matrix + np.triu(full_matrix, 1).T


    @property
    def dim(self)->int:
        '''Matrix dimension'''
        return self._matrix_dim

    @property
    def trace(self)->float:
        return self._trace

    @property
    def eigenvalues(self)->NDArray:
        '''Matrix eigenvalues, if called first will perform eigen decomposition'''
        if self._eigenvalues is None:
            self.perform_eigen_decomposition()

        return self._eigenvalues

    @property
    def norm(self)->float:
        '''Frobenius norm of the matrix'''
        return self._matrix_norm

    @property
    def data(self)->NDArray:
        '''
        Returns copy of underlying matrix
        '''
        return self._matrix.copy()

    def __getitem__(self, data):
        '''
        Allows user to directly interface with underlying numpy array
        '''
        return self._matrix.__getitem__(data)

    def perform_eigen_decomposition(self)->Tuple[NDArray, NDArray]:
        '''
        Performs eigen value decomposition with a more verbose error
        '''
        try:
            self._eigenvalues = eigvalsh(self._matrix)
        except np.linalg.LinAlgError as e:
            raise EigenDecompError(f"Cannot decompose matrix {self._name}") from e

        return self._eigenvalues

    def perform_cholesky_decompositon(self)->'RootMatrix':
        chol_matrix = np.linalg.cholesky(self._matrix)
        self._cholesky_decomp = RootMatrix(chol_matrix, f"{self._name}_chol")
        return self._cholesky_decomp

    @property
    def cholesky_component(self)->'RootMatrix':
        '''
        Lazy evaluation of cholesky decompositon. Ensure matrix is always connected to cholesky decomp.
        '''
        if self._cholesky_decomp is None:
            self.perform_cholesky_decompositon()

        return self._cholesky_decomp

    @property
    def inverse(self)->'RootMatrix':
        inverse = np.linalg.inv(self._matrix)
        return RootMatrix(inverse)

    @property
    def name(self)->str:
        return self._name

    # Left multiply
    def __mul__(self, other: 'float | RootMatrix'):
        if isinstance(other, int):
            mat = other*self._matrix
        elif isinstance(other, RootMatrix):
            mat = np.matmul(self._matrix, other._matrix)
        else:
            raise ValueError(f"Cannot mutliply RootMatrix by type {type(other)}")
        
        return RootMatrix(mat, self._name)



    def __rmult__(self, other):
        if isinstance(other, int):
            mat = other*self._matrix
        elif isinstance(other, RootMatrix):
            mat = np.matmul(other._matrix, self._matrix)
        else:
            raise ValueError(f"Cannot mutliply RootMatrix by type {type(other)}")

        return RootMatrix(mat, self._name)

    def __eq__(self, other):
        if not isinstance(other, RootMatrix):
            return False
        
        return np.array_equal(self._matrix, other._matrix)

# ------------------------
class MatrixFileHandler:
# ------------------------
    def __init__(self, root_file_path: Path, matrix_stem: str, index_range: Tuple[int, int, int]):
        logging.info(f"[green]Loading matrices from [/][bold blue]{root_file_path}[/][green] with stem [/][bold blue]{matrix_stem}[/]")

        with uproot.open(root_file_path) as root_file:
            self._matrix_list = [RootMatrix.from_root_file(root_file, f"{i}_{matrix_stem}") for i in tqdm(range(index_range[0], index_range[1], index_range[2]), f"Loading matrices from {root_file_path}")]

    @property
    def matrix_list(self)->List[RootMatrix]:
        return self._matrix_list
