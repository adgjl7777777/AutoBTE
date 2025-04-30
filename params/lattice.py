from AutoBTE.optimizer.base import vasp_run  # Import specific function
from ase import Atoms
import numpy as np
from copy import deepcopy
import os
import matplotlib.pyplot as plt
from ase.calculators import calculator

def find_lattice(structure: Atoms, directory: str, cores: int = 1, k_points: list[int] = (1,1,1), E_cut: float = 600) -> None:
    """Find the best lattice parameters for VASP calculation.

    Args:
        structure (ase.Atoms): Structure information including periodic boundary conditions (pbc).
        directory (str): Calculation save directory.
        cores (int, optional): Number of CPU/GPU cores. Defaults to 1.
        k_points (bool, optional): Optimized k points calculated by kpts.py
        E_cut (float, optional): Optimized Encut calculated by Encut.py

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
    
    scale_potential = []
    for scale_factor in np.arange(0.9, 1.1 + 0.005, 0.005):
        # Get the current cell
        struct_target = deepcopy(structure)
        original_cell = struct_target.get_cell()

        # Scale the cell by multiplying with the factor
        new_cell = original_cell * scale_factor

        # Update the structure with the new cell
        struct_target.set_cell(new_cell, scale_atoms=True)  # `scale_atoms=True` moves atoms accordingly
        abc = struct_target.get_cell_lengths_and_angles()
        try:
            struct_target, energy = vasp_run(struct_target,run_type="single",k_point=k_points, cores=cores, directory=directory, cell_opt=False, E_cut=E_cut)
            scale_potential.append([max(abc[0:3]),energy,abc[0],abc[1],abc[2],abc[3],abc[4],abc[5]])
        except calculator.CalculationFailed:
            print(f"Warning: VASP calculation failed for lattice_parameter={max(abc[0:3])}. Skipping this value.")
            scale_potential.append([max(abc[0:3]),99999,abc[0],abc[1],abc[2],abc[3],abc[4],abc[5]])  # None if calculation failed
    
        

    # Save results to file
    with open(os.path.join(directory, "lattice_result.txt"), "w") as f:
        for energy in scale_potential:
            f.write(f"({energy[2]:.6f},{energy[3]:.6f},{energy[4]:.6f},{energy[5]:.6f},{energy[6]:.6f},{energy[7]:.6f}): {energy[1]:.6f} eV\n")
        indices = np.argsort(np.array(scale_potential)[:, 1])[:5]
        f.write("=========================\n")
        f.write("Valid Lattice Parameters: \n")
        for i in indices:
            energy = scale_potential[i]
            f.write(f"({energy[2]:.6f},{energy[3]:.6f},{energy[4]:.6f},{energy[5]:.6f},{energy[6]:.6f},{energy[7]:.6f}): {energy[1]:.6f} eV\n")

    plotscale = np.array(scale_potential)
    # Plot the data
    plt.plot(plotscale[:,0], plotscale[:,1])
    plt.xlabel("Lattice Length (Å)")
    plt.ylabel("Potential Energy (eV)")
    plt.title("Lattice Optimization: Potential Energy vs Lattice Length")
    plt.grid(True)
    plt.savefig(os.path.join(directory,"lattice.png"),dpi=600)
    plt.clf()
    plt.cla()
    plt.close()