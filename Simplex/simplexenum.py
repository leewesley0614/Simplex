from enum import Enum

class SimplexState(Enum):
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUND = 3

class MatrixState(Enum):
    COMMON = 1
    BASE = 2
    NONBASE = 3