import numpy as np
from simplexenum import MatrixState

class SimplexArray(np.ndarray):
    
    def __new__(cls, input_array = None, state:MatrixState = None):
        '''
        创建SimplexArray对象，继承np.ndarray的所有功能
        '''

        obj = np.asarray(input_array).view(cls)
        # 添加属性
        obj.state_ = state
        return obj
    
    def __init__(self, input_array = None, state:MatrixState = MatrixState.COMMON):
        self.state_ = state


    def GetColumns(self, colindexs):
        if self is None: raise ValueError("Matrix is not set.")
        if any(idx < 0 or idx >= self.shape[1] for idx in colindexs):
            raise IndexError(f"One or more column indices are out of bounds for matrix with {self.shape[1]} columns.")
        return self[:, colindexs]
    
    def IsBase(self):
        if self.state_ == MatrixState.BASE: return True
        else: return False
    
    