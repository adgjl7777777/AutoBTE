from AutoBTE.optimizer.base import vasp_run  # Import specific function
from ase import Atoms
import numpy as np
from copy import deepcopy
import os
import matplotlib.pyplot as plt
from ase.calculators import calculator
def find_k(structure: Atoms, directory: str, cores: int = 1, ratio: list[float] = (-1, -1, -1), cell_opt: bool = True) -> None:
    """Find the best k points for VASP calculation.

    Args:
        structure (ase.Atoms): Structure information including periodic boundary conditions (pbc).
        directory (str): Calculation save directory.
        cores (int, optional): Number of CPU/GPU cores. Defaults to 1.
        ratio (list[float], optional): k points ratio for customizing.
        cell_opt (bool, optional): True if cell shape should be optimized.

    Raises:
        ValueError: If the structure has zero atoms.

    Returns:
        None
    """
    ###########################
    criteria = 0.01
    count = 3
    E_cut = 600  # Safe selection
    ###########################

    struct_target = deepcopy(structure)

    # Raise an error if the structure is empty
    if len(struct_target) == 0:
        raise ValueError("Error: The provided structure has zero atoms. Please provide a valid structure.")

    # If ratio is (-1, -1, -1), automatically determine it
    if ratio == (-1, -1, -1):
        ratio = np.array(struct_target.get_cell_lengths_and_angles()[:3])
        if max(ratio) == 0:
            ratio = np.ones(3)  # Set to at least 1 to prevent errors
    ratio = ratio / max(ratio)

    os.makedirs(directory, exist_ok=True)

    potential = []
    k_ans = []
    k_max = 20 # Prevent infinite loop
    for kk in range(1, k_max+1):
        k = np.ceil(kk * ratio).astype(int)
        try:
            struct_target, energy = vasp_run(struct_target,run_type="geo_opt",k_point=k, cores=cores, directory=directory, cell_opt=cell_opt, E_cut=E_cut)
            potential.append(energy)
        except calculator.CalculationFailed:
            print(f"Warning: VASP calculation failed for k_points={k}. Skipping this value.")
            potential.append(None)  # None if calculation failed
    # Find valid values
    for kk in range(1, k_max+1):
        if abs(potential[kk-1] - potential[-1]) < criteria * len(struct_target):
            k_ans.append(np.ceil(kk * ratio).astype(int))

    # Save results to file
    with open(os.path.join(directory, "k_point_result.txt"), "w") as f:
        plotter = []
        for i, energy in enumerate(potential):
            if energy is None:
                f.write(f"{i+1}: Calculation Failed\n")
            else:
                f.write(f"{i+1}: {energy:.6f} eV\n")
                plotter.append([i+1,energy])
        f.write("=========================\n")
        f.write("Valid k: \n")
        if k_ans:
            for i in k_ans:
                f.write("["+", ".join(map(str, np.ceil(i * ratio)))+"]")  # Output list in a single line
                f.write("\n")
        else:
            f.write("None (No valid Encut found)")
            f.write("\n")
            # Plot the data
    if len(plotter)>0:
        plotter = np.array(plotter)
        plt.plot(plotter[:,0], plotter[:,1])
        plt.xlabel("k Points")
        plt.ylabel("Potential Energy (eV)")
        plt.title("k Points Optimization: Potential Energy vs k points")
        plt.grid(True)
        plt.savefig(os.path.join(directory,"k.png"),dpi=600)
    plt.clf()
    plt.cla()
    plt.close()