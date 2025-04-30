import os
import sys

from ase import Atoms
from ase.build import bulk
import AutoBTE
print(dir(AutoBTE))
# 알루미늄(Al) FCC 구조의 최소 격자 생성
al_lattice = bulk("Al", "fcc", a=4.05)  # a = 4.05 Å (실제 실험값)

# ASE에서 생성한 격자 정보 출력
print("Lattice parameters (Å):", al_lattice.get_cell().lengths())
print("Atomic positions:")
print(al_lattice)

# 파일로 저장 (VASP POSCAR 형식)
al_lattice.write("test/test.xyz", format="extxyz")
#AutoBTE.params.kpts.find_k(al_lattice, "./test/k/", cores = 4, cell_opt = True)
#AutoBTE.params.Encut.find_e(al_lattice, "./test/e/", cores = 4, cell_opt = True,k_points=(6,6,6))
AutoBTE.params.lattice.find_lattice(al_lattice, "./test/l/", cores = 4,k_points=(6,6,6),E_cut=300)