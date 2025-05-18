from AutoBTE.optimizer.base import vasp_run  # Import specific function
from ase import Atoms
import numpy as np
from copy import deepcopy
import os
import matplotlib.pyplot as plt
from ase.calculators import calculator
def find_e(structure: Atoms, directory: str, cores: int = 1, cell_opt: bool = True, k_points: list[int] = (1,1,1)) -> None:
    """Find the best Energy cutoff for VASP calculation.

    Args:
        structure (ase.Atoms): Structure information including periodic boundary conditions (pbc).
        directory (str): Calculation save directory.
        cores (int, optional): Number of CPU/GPU cores. Defaults to 1.
        cell_opt (bool, optional): True if cell shape should be optimized.
        k_points (bool, optional): Optimized k points calculated by kpts.py

    Raises:
        ValueError: If the structure has zero atoms.

    Returns:
        None
    """
    ###########################
    criteria = 0.01
    count = 3
    ###########################

    struct_target = deepcopy(structure)

    # Raise an error if the structure is empty
    if len(struct_target) == 0:
        raise ValueError("Error: The provided structure has zero atoms. Please provide a valid structure.")

    os.makedirs(directory, exist_ok=True)

    potential = []
    e_ans = []
    e_max = 2000 # Prevent infinite loop
    for ee in range(100,e_max+100,50):
        try:
            struct_target,energy = vasp_run(struct_target,run_type="geo_opt",k_point=k_points, cores=cores, directory=directory, cell_opt=cell_opt, E_cut=ee)
            potential.append(energy)
        except calculator.CalculationFailed:
            print(f"Warning: VASP calculation failed for E_cut={ee}. Skipping this value.")
            potential.append(None)  # None if calculation failed

    # Find valid values
    for e in range(len(potential)):
        if potential[e] is not None and abs(potential[e] - potential[-1]) < criteria * len(struct_target):
            e_ans.append(100 + e * 50)

    # Save results to file
    with open(os.path.join(directory, "encut_result.txt"), "w") as f:
        plotter = []
        for e, energy in enumerate(potential):
            if energy is None:
                f.write(f"{100 + 50 * e}: Calculation Failed\n")
            else:
                f.write(f"{100 + 50 * e}: {energy:.6f} eV\n")
                plotter.append([100 + 50 * e,energy])
        f.write("=========================\n")
        f.write("Valid Encut: ")
        if e_ans:
            f.write(", ".join(map(str, e_ans)))  # Output list in a single line
        else:
            f.write("None (No valid Encut found)")
        f.write("\n")
    if len(plotter)>0:
        plotter = np.array(plotter)
        plt.plot(plotter[:,0], plotter[:,1])
        plt.xlabel("Energy Cutoff (eV)")
        plt.ylabel("Potential Energy (eV)")
        plt.title("Energy Cutoff Optimization: Potential Energy vs Energy Cutoff")
        plt.grid(True)
        plt.savefig(os.path.join(directory,"e.png"),dpi=600)
    plt.clf()
    plt.cla()
    plt.close()