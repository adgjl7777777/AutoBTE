# AutoBTE/__init__.py

"""
AutoBTE: A high-level interface for your Boltzmann-transport
and DFT-based workflows.
"""

# 버전 정보
__version__ = "0.2.0"
# BoltzTrap 하위모듈에서 BTE 관련 클래스나 함수 가져오기
from .BoltzTrap.BTE import btp2   # 예시 이름. 실제 모듈명에 맞게 수정

# optimizer 서브패키지 노출
from .optimizer.base import Optimizer
from .optimizer.base import emt_run, vasp_run, chgnet_run

# params 서브패키지 노출
from .params.Encut import find_e
from .params.kpts import find_k
from .params.lattice import find_lattice

# __all__ 은 `from AutoBTE import *` 할 때 노출될 이름들
__all__ = [
    "btp2",
    "Optimizer", "emt_run", "vasp_run", "chgnet_run",
    "find_e", "find_k", "find_lattice",
    "__version__",
]
