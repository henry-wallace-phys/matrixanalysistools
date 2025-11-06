'''
Class containing convergence analysis of a single matrix
'''

from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional, Any

import numpy as np
from numpy.typing import NDArray
from tqdm.rich import tqdm

from matrixanalysistools.matrix_handler.root_matrix import RootMatrix, MatrixFileHandler
from matrixanalysistools.exceptions import EigenDecompError

# ------------------------
class ConvergenceAnalyser:
# ------------------------
    '''
    Analyses rate of change of eigenvalues from an (assumed) ordered list of RootMatrices from some process
    '''

    def __init__(self, matrix_list: List[RootMatrix] | MatrixFileHandler):

        if isinstance(matrix_list, MatrixFileHandler):
            self._matrix_list = matrix_list.matrix_list
        else:
            self._matrix_list = matrix_list


        self._eigenvalues: Optional[NDArray] = None
        self._eigenvalue_roc: Optional[NDArray] = None

        self._cholesky_components: Optional[NDArray] = None
        self._suboptimality: Optional[NDArray] = None

        self.eigen_decompose()
        
    def cholesky_decompose(self, n_workers: Optional[int]=None):
        '''
        Just does the eigen decomposition for everything in a "nice" multithreaded way
        '''
        self.__decompose(self._matrix_list, 'perform_cholesky_decompositon', n_workers)
        # Bit inefficient to do this twice but it's easier to do it here since the TPE just caches
        self._cholesky_components = np.array([m.cholesky_component for m in self._matrix_list])
        return self._cholesky_components

    def eigen_decompose(self, n_workers: Optional[int]=None):
        self.__decompose(self._matrix_list, 'perform_eigen_decomposition', n_workers)
        # Bit inefficient to do this twice but it's easier to do it here since the TPE just caches
        self._eigenvalues = np.array([m.eigenvalues for m in self._matrix_list])
        return self._eigenvalues

    @classmethod
    def __decompose(cls, iterable: List[Any], method: str, n_workers: Optional[int]=None):
        
        if not hasattr(iterable[0], method):
            raise AttributeError(f"Error, {method} not a method in RootMatrix" )
        
        with ThreadPoolExecutor(n_workers) as tpe:
            futures = [tpe.submit(getattr(m, method)) for m in iterable]

            for future in tqdm(as_completed(futures), total=len(futures), desc=f"Performing {method} on input matrices"):
                future.result()

    @property
    def matrix_norms(self)->List[float]:
        return [m.norm for m in self._matrix_list]
    
    @property
    def cholesky_components(self)->NDArray:
        if self._cholesky_components is None:
            self.cholesky_decompose()
        return self._cholesky_components

    def __get_relative_change(self, i: int)->float:
        if self._eigenvalues is None:
            raise EigenDecompError("Matrices haven't been decomposed!")

        if i<1 or i>len(self._eigenvalues):
            raise ValueError(f"Index {i} outside of range (1, {len(self._eigenvalues)})")

        # old_norm: float = np.linalg.norm(self._eigenvalues[i-1])
        change: float = np.linalg.norm(self._eigenvalues[i]-self._eigenvalues[i-1])
        return change

    def calc_eigenvalue_roc(self, n_workers: Optional[int]=None)->NDArray:
        # Get the eigenvalues
        with ThreadPoolExecutor(n_workers) as tpe:
            futures = [tpe.submit(self.__get_relative_change, i) for i in range(1, len(self._matrix_list)-1)]
            self._eigenvalue_roc = np.array([future.result() for future in futures])

        return self._eigenvalue_roc

    @property
    def eigenvalues(self):
        return self._eigenvalues

    @property
    def eigenvalue_roc(self)->NDArray:
        if self._eigenvalue_roc is None:
            return self.calc_eigenvalue_roc()

        return self._eigenvalue_roc
    
    def __eigen_suboptimality(self, eigen_vals: NDArray):
        numerator = sum(eigen_vals**-2)
        denominator = sum(eigen_vals**-1)**2
        
        return len(eigen_vals)*numerator/denominator
    
    def calc_suboptimality(self):
        '''
        From https://probability.ca/jeff/ftpdir/adapteveryoneprnt.pdf
        '''
        # Get the "final" ~converged matrix
        end_matrix = self.cholesky_components[-1]
        
        # Now we need chol[i]*converged
        mat_vec = ConvergenceAnalyser([m*end_matrix for m in self.cholesky_components])
        
        # Get the eigenvalues
        eigen_vec  = mat_vec.eigenvalues
        
        # Get suboptimality
        self._suboptimality = np.array([self.__eigen_suboptimality(e) for e in eigen_vec])
        return self._suboptimality
    
    @property
    def suboptimality(self):
        if self._suboptimality is None:
            self.calc_suboptimality()
            
        return self._suboptimality