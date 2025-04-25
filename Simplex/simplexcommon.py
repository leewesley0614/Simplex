import numpy as np
def GetColumn(mat:np.ndarray, colindex:int):
    if mat is None:
        raise ValueError("Matrix is not set.")
    if (colindex <0 or int > mat.shape[1]):
        raise IndexError(f"Column index {colindex} is out of bounds for matrix with {mat.shape[1]}")
    return mat[:, colindex]