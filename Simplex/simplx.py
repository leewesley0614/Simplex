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
        self.nonbasecolidx_:list = None
        self.state_: SimplexState = None
        self.obj_ = None        # objective funciton
        self.B_ = None # base matrix
        self.N_ = None # non-base matrix
        self.cb_ = None # base cost coefficient
        self.cn_ = None # non-base cost coefficient
        self.xb_ = None # base var
        self.xn_ = None # non-base var
        self.SetBase() # set base
        self.SetNonBase() # set non base
    
    def SetBase(self):
        self.B_ = self.A_[:, self.basecolidx_]
        self.cb_ = self.c_[self.basecolidx_]
    
    def SetNonBase(self):
        col_idx = list(range(self.A_.shape[1]))
        self.nonbasecolidx_ = [ Item for Item in col_idx if Item not in self.basecolidx_]
        self.N_ = self.A_[:, self.nonbasecolidx_]
        self.cn_ = self.c_[self.nonbasecolidx_]
    

    def ShowBase(self):
        print(f"CURRENT BASE VAR: x{self.basecolidx_}\n",
              f"CURRENT BAER MATRIX:\n{self.B_}\n",
              f"CURRENT THE VALUE OF BASE COST:{self.cb_}\n",
              f"CURRENT THE VALUE OF BASE VARIABLE:{self.xb_}")
    def ShowNonBase(self):
        print(f"CURRENT BASE VAR: x{self.nonbasecolidx_}\n",
              f"CURRENT NON BAER MATRIX:\n{self.N_}\n",
              f"CURRENT THE VALUE OF NON BASE COST:{self.cn_}\n")

    def CalReducedCost(self): # calculate reduced cost
        return self.cb_ @ np.linalg.solve(self.B_, self.N_) - self.cn_
    
    
    def Swap(self, in_base_idx, out_base_idx):
        # 交换变量
        in_var = self.basecolidx_[out_base_idx]
        self.basecolidx_[out_base_idx] = self.nonbasecolidx_[in_base_idx]
        self.nonbasecolidx_[in_base_idx] = in_var
        # 交换列
        in_vec = self.B_[:, out_base_idx].copy()
        self.B_[:, out_base_idx] = self.N_[:, in_base_idx].copy()
        self.N_[:, in_base_idx] = in_vec
        # self.B_ = self.A_[:, self.basecolidx_]
        # self.N_ = self.A_[:, self.nonbasecolidx_]
        # 交换成本系数
        in_cost = self.cb_[out_base_idx]
        self.cb_[out_base_idx] = self.cn_[in_base_idx]
        self.cn_[in_base_idx] = in_cost

    def Run(self):
        tolerance = 1e-9
        while(self.state_ != SimplexState.OPTIMAL):
            self.xb_ = np.linalg.solve(self.B_, self.b_)
            self.xn_ = np.zeros(len(self.nonbasecolidx_))
            self.obj_ = self.cb_ @ self.xb_ # calculate objective function
            self.ShowBase()
            self.ShowNonBase()
            reduced_cost = self.CalReducedCost() # calculate the reduced cost of non-base var
            print(f"REDUCED COST: {reduced_cost}\n")
            if(reduced_cost.max() <= tolerance): # 最大检验数小于等于0，现行基本可行解为最优解
                self.state_ = SimplexState.OPTIMAL
            else:
                in_base_idx = np.argmax(reduced_cost) # 入基变量索引
                pk = self.N_[:, in_base_idx] # 入基变量对应的列系数
                yk = np.linalg.solve(self.B_, pk)
                if(yk.max() <= tolerance):
                    self.state_ =SimplexState.UNBOUND
                    break
                else:
                    # 记录yk > 0 的索引
                    pos_yk_idx = np.where(yk > tolerance)[0]
                    out_base_idx_script = np.argmin(self.xb_[pos_yk_idx] /  yk[pos_yk_idx])
                    out_base_idx = pos_yk_idx[out_base_idx_script] # 出基变量索引
                    self.Swap(in_base_idx, out_base_idx) # 换基
        match self.state_:
            case(SimplexState.OPTIMAL):
                print("Simplex State: OPTIMAL\n",
                      f"BASE VAR INDEX:x_{self.basecolidx_}\n",
                      f"BASE VAR VALUE:{self.xb_}\n",
                      f"OPTIMAL VALUE: {self.obj_}")
            case(SimplexState.UNBOUND):
                print("Simplex State: UNBOUND")







if __name__ == "__main__":
    ## Test Data1

    # test_data = [[-1, 2, 1, 0, 0],
    #              [2, 3, 0, 1, 0],
    #              [1, -1, 0, 0, 1]]
    # c = [-4, -1, 0, 0, 0]
    # b = [4, 12, 3]
    # basecolidx = [2, 3, 4]

    ## Test Data2
    # test_data = [[1, 1, -2, 1, 0, 0],
    #              [2, -1, 4, 0, 1, 0],
    #              [-1, 2, -4, 0, 0, 1]]
    # c = [1, -2, 1, 0, 0, 0]
    # b = [10, 8, 4]
    # basecolidx = [3, 4, 5]
    ## Test Data3
    test_data = [[1, 1, 2, 1, 0],
                 [1, 4, -1, 0, 1]]
    c = [-2, -1, 1, 0, 0]
    b = [6, 4]
    basecolidx = [3, 4]
    ## Initialize, 初始化
    simplex = Simplex(SimplexArray(test_data), np.array(b), np.array(c), basecolidx)
    simplex.Run()
    

    
