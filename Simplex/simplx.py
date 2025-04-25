# Simplex Implement
import numpy as np
# from simplexcommon import GetColumn
from simplexenum import SimplexState, MatrixState
from simplexarray import SimplexArray

class SimplexException(Exception):
    def __init__(self, message = "Simplex 算法异常"):
        super().__init__(message)


class Simplex:
    ## data member
    def __init__(self, A:SimplexArray, b, c, basecolidx):
        self.A_:SimplexArray = A   # A_ is coefficient matrix
        self.b_:np.ndarray = b      # b_ is resource vector
        self.c_:np.ndarray = c      # c_ is cost coefficient
        self.basecolidx_:list = basecolidx   # base column index
        self.x_:np.ndarray = None
        self.state_: SimplexState = None
        self.obj_ = None        # objective funciton
        self.B_ = None
        self.N_ = None
        self.cb_ = None
        self.cn_ = None
        self.xb_ = None
        self.xn_ = None

    def GetBase(self):
        B = self.A_[:, self.basecolidx_]
        c_b = self.c_[self.basecolidx_]
        return B, c_b
    
    def GetNonBase(self):
        col_idx = list(range(self.A_.shape[1]))
        nonbasecolidx = [Item for Item in col_idx if Item not in self.basecolidx_]
        N = self.A_[:, nonbasecolidx]
        c_n = self.c_[nonbasecolidx]
        return N, c_n
    
    
    '''Get IN Base idx'''
    def GetInBaseIdx(self): 
        reduced_cost = self.GetReducedCost() # get reduced cost
        max_reduced_cost = reduced_cost.max() # get maximum reduced cost
        if(max_reduced_cost <= 0 ): 
            self.state_ = SimplexState.OPTIMAL
            return None
        else:
            in_base_idx = np.argmax(max_reduced_cost) # get in base variable index subscript, for N_
            return in_base_idx
    
    '''Get In Base Column'''
    def GetInBaseCol(self):
        in_base_idx = self.GetInBaseIdx()
        return self.N_[:, in_base_idx]
    
    '''Get InBase Var'''
    def GetInBaseVar(self):
        in_base_idx = self.GetInBaseIdx()
        return self.xn_[in_base_idx]

    def GetOutBaseIdx(self):
        pk = self.GetInBaseCol()
        yk = np.linalg.inv(self.B_) @ pk
        bool_mask = yk > 0 # 判断哪些yk小于0
        pos_yk = yk[bool_mask]
        pos_yk_idx = np.where(bool_mask)[0]

        if(pos_yk.size()==0): 
            self.state_ = SimplexState.UNBOUND
            raise SimplexException("问题不存在有限最优解")
        
        b_bar = np.linalg.inv(self.B_) @ self.b_
        xk_val = np.min(b_bar[:, pos_yk_idx] / pos_yk)
        out_base_idx = pos_yk_idx[np.argmin(b_bar[:, pos_yk_idx] / pos_yk)]
        return out_base_idx, xk_val
    
    def GetOutBaseCol(self):
        out_base_idx, _ = self.GetOutBaseIdx()
        return self.B_[:, out_base_idx]
    
    def GetOutBaseVar(self):
        out_base_idx, _ = self.GetOutBaseIdx()
        return self.xb_[out_base_idx]


    def CalReducedCost(self): # calculate reduced cost
        return self.cb_ @ np.linalg.inv(self.B_) @ self.N_ - self.cn_
    
    
    def Swap(self, in_base_idx, out_base_idx):
        # 交换变量
        in_var = self.basecolidx_[out_base_idx]
        self.basecolidx_[out_base_idx] = self.nonbasecolidx_[in_base_idx]
        self.nonbasecolidx_[in_base_idx] = in_var
        # 交换列
        self.SetBase(self.basecolidx_)
        self.SetNonBase(self.nonbasecolidx_)

        
    

    
    
    def Run(self):
        self.xb_




if __name__ == "__main__":
    ## Test Data
    test_data = [[-1, 2, 1, 0, 0],
                 [2, 3, 0, 1, 0],
                 [1, -1, 0, 0, 1]]
    c = [-4, -1, 0, 0, 0]
    b = [4, 12, 3]
    basecolidx = [2, 3, 4]
    ## Initialize, 初始化
    simplex = Simplex(SimplexArray(test_data), np.array(b), np.array(c), basecolidx)
    cur_base, cur_cb = simplex.GetBase()
    cur_nonbase, cur_cn = simplex.GetNonBase()
    print(f"Current Base Matrix: \n {cur_base} \n",
          f"Current cb: {cur_cb} \n")
    
    print(f"Current NonBase Matrix: \n {cur_nonbase} \n",
          f"Current cn: {cur_cn} \n")
    

    
