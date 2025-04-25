from enum import Enum

class SimplexState(Enum):
    OPTIMAL = 1
    UNBOUND = 2

class MatrixState(Enum):
    COMMON = 1
    BASE = 2
    NONBASE = 3