import numpy as np

# Exceptions
class ObjectNotFoundError(Exception):
    '''When object isn't found in a root file'''
    ...

class NonSquareMatrixError(Exception):
    '''For non-square matrices'''
    ...

class EigenDecompError(np.linalg.LinAlgError):
    '''Wraps around np linalg for a more verbsose error'''
    ...
