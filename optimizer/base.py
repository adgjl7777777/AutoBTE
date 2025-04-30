from ase import Atoms
from pymatgen.io.ase import AseAtomsAdaptor
from pymatgen.core import Structure
from typing import List, Callable, Union
from ase.calculators.vasp import Vasp
from copy import deepcopy

from ase.calculators.emt import EMT
from ase.optimize import BFGS, BFGSLineSearch
from ase.constraints import UnitCellFilter
from ase.md.npt import NPT
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution, Stationary
from ase import units

from chgnet.model import StructOptimizer
from chgnet.model.model import CHGNet
from chgnet.model.dynamics import MolecularDynamics

import os


def structure_converter(structure):
    calc_structure=None
    if isinstance(structure, Atoms):
        calc_structure=deepcopy(structure)
    elif hasattr(structure, "to") and "pymatgen" in str(type(structure)):
        calc_structure = AseAtomsAdaptor.get_atoms(structure)
    else:
        raise TypeError("The structure must be an ASE Atoms or Pymatgen Structure object.")
    return calc_structure
def compute_ramp_t(a: float, t1: float, t2: float, ramp_type: str="default") -> float:
    """
    Compute ramp t value based on a given a in the range [0, 1].

    Args:
        a (float): Input value in the range [0,1].
        t1 (float): Initial value of t.
        t2 (float): Final value of t.
        ramp_type (str): Ramp type.

    Returns:
        float: Computed t value.
    """
    if a < 0 or a > 1:
        raise ValueError("a must be in the range [0, 1]")
    if ramp_type=="default":
        if a <= 0.25:
            return t1
        elif a <= 0.75:
            return t1 + (t2 - t1) * ((a - 0.25) / 0.5)  # Linear interpolation
        else:
            return t2
    elif ramp_type=="short":
        if a <= 1/3:
            return t1
        else:
            return t1 + (t2 - t1) * ((a - 1/3) / (2/3))  # Linear interpolation
    
def vasp_run(structure,run_type="single",system_name="SINGLE",k_point=(1,1,1),E_cut=600,cores=1,lreal="Auto",directory="./",idipol=0,cell_opt=True,new_start=True,Ti=298,Tf=298,dt=1,steps=1000, vasp_kwargs=None):
    # Convert Pymatgen structure to ASE Atoms if necessary
    calc_structure=structure_converter(structure)
    vasp_kwargs = vasp_kwargs or {}
    
    vasp_params = {
        "system": system_name,  # vasp project name

        "istart": 0 if new_start else 2,  # not reading WAVECAR file (new start)
        "icharg": 2 if new_start else 1,  # not reading CHGCAR file (new start)

        "kpts": k_point,  # k points
        "encut": E_cut,  # energy cutoff
        "xc": "pbe",  # functional settings (PBE)
        "gga": "pe",  # functional settings (PBE)
        "ediff": 1e-6,  # energy criteria (1e-6 eV)
        "ispin": 1,  # without spin polarization

        "lorbit": 11,  # projection method
        "ivdw": 11,  # vdW (dispersion) correction (DFT-D3)
        "algo": "Normal",  # electronic minimization algorithm
        "amin": 0.01,  # minimal mixing parameter in Kerker's initial approximation
        "ismear": 0,  # smearing method
        "sigma": 0.03,  # width of the smearing

        "nelm": 300,  # maximum number of electronic SC steps
        "ncore": cores,  # number of CPU/GPUs

        "lreal": lreal,  # if the structure size is small, just use lreal=False

        "directory": os.path.join(directory,"vasp/"),  # VASP directory
    }
    if idipol != 0:
        vasp_params["idipol"] = idipol
    if run_type == "single":
        _=1
    elif run_type=="geo_opt":
        vasp_params["ediffg"] = 1e-2 # force criteria (0.01 eV/Å per atom)
        vasp_params["ibrion"] = 2  # structure optimization method (2: Conjugate Gradient, 3: Quasi-Newton)
        vasp_params["isif"] = 3 if cell_opt else 2  # atoms & cell optimization (isif=2 for only atoms)
        vasp_params["nsw"] = 200  # maximum number of ionic steps
    elif run_type=="NPT":
        vasp_params["mdalgo"] = 3 # NPT
        vasp_params["ibrion"] = 0  # MD
        vasp_params["isif"] = 3 # atoms & cell optimization
        vasp_params["smass"] = -1 # controlling the velocities during an ab-initio molecular-dynamics run
        vasp_params["pmass"] = 1000 # the fictitious mass of the lattice degrees of freedom
        vasp_params["langevin_gamma"] = (10,10,10) # Langevin friction coefficient for three atomic species
        vasp_params["langevin_gamma_l"] = 10 # Langevin friction coefficient for lattice degrees of freedom

        vasp_params["tebeg"] = Ti  # T start
        vasp_params["teend"] = Tf  # T end
        vasp_params["potim"] = dt  # time step
        vasp_params["nsw"] = steps  # total steps

        vasp_params["lwave"] = False  # do not write the wavefunction
        vasp_params["lcharg"] = False  # do not write the charge densities
    else:
        raise ValueError("Invalid run_type. Choose from 'single', 'geo_opt', or 'NPT'.")
    vasp_params.update(vasp_kwargs)
    calc_structure.calc=Vasp(**vasp_params)
    energy = calc_structure.get_potential_energy()
    return calc_structure,energy


def emt_run(structure,run_type="single",directory="./",cell_opt=True,Ti=298,Tf=298,dt=1,steps=1000):
    calc_structure=structure_converter(structure)
    save_dir =os.path.join(directory,"emt")
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    
    calc_structure.calc = EMT()  # Set EMT calculator
    energy = 0
    if run_type == "single":
        # Single-point energy calculation
        energy = calc_structure.get_potential_energy()

    elif run_type == "geo_opt":
        # Geometry optimization
        
        if cell_opt:
            # Optimize both atomic positions and unit cell
            calc_structure = UnitCellFilter(calc_structure)  # Apply cell optimization filter
            optimizer = BFGSLineSearch(
                calc_structure,
                trajectory=os.path.join(save_dir, "geo_opt.traj"),
                logfile=os.path.join(save_dir, "geo_opt.log")
        )  # BFGS for cell optimization
        else:
            # Optimize only atomic positions
            optimizer = BFGS(
                calc_structure,
                trajectory=os.path.join(save_dir, "geo_opt.traj"),
                logfile=os.path.join(save_dir, "geo_opt.log")
            )
        optimizer.run(fmax=0.01)  # Convergence criterion
        energy = calc_structure.get_potential_energy()

    elif run_type == "NPT":
        # NPT Molecular Dynamics
        unit_dt= dt * units.fs, 1000  # Time step and number of steps

        # Set initial velocity distribution
        MaxwellBoltzmannDistribution(calc_structure, temperature_K=Ti)
        Stationary(calc_structure)  # Remove center-of-mass motion

        # Define NPT dynamics
        dyn = NPT(
            calc_structure,
            timestep=unit_dt,
            temperature=Tf,
            externalstress=0.0001013,  # Atmospheric pressure
            ttime=25 * units.fs,  # Thermostat time constant
            pfactor=100.0,  # Barostat parameter
            trajectory=os.path.join(save_dir, "npt.traj"),
            logfile=os.path.join(save_dir, "npt.log"),
        )

        dyn.run(steps)
        energy = calc_structure.get_potential_energy()

    else:
        raise ValueError("Invalid run_type. Choose from 'single', 'geo_opt', or 'NPT'.")

    return calc_structure, energy  # Return the updated structure

def chgnet_run(structure,run_type="geo_opt",directory="./",cell_opt=True,new_start=True,Ti=298,Tf=298,dt=1,steps=1000,ramp_type="default"):
    calc_structure=structure_converter(structure)
    save_dir =os.path.join(directory,"chgnet")
    os.makedirs(save_dir, exist_ok=True)  # Ensure directory exists
    
    calc_structure.calc = EMT()  # Set EMT calculator
    energy = 0
    if run_type == "geo_opt":
        relaxer = StructOptimizer()
        relax_result = relaxer.relax(calc_structure, fmax=0.01, save_path=os.path.join(save_dir, "geo_opt.traj"), verbose=False)
        energy = calc_structure.get_potential_energy()
        
    elif run_type == "NPT":
                
        # NPT 시뮬레이션 생성
        md = MolecularDynamics(
            atoms=calc_structure,
            ensemble="npt",
            temperature=Ti, 
            timestep= dt,
            logfile=os.path.join(save_dir, "npt.log"),
            loginterval= int(10 * dt),
            trajectory=os.path.join(save_dir, "npt.traj"),
        )

        def temperature_ramp(step):
            ramp_temp = compute_ramp_t(step / steps,Ti,Tf,ramp_type=ramp_type)
            md.dyn.set_temperature(ramp_temp)
            return ramp_temp

        temp_interval = 1000

        # Attach the temperature ramp function (1ps 간격으로 온도 업데이트)
        md.dyn.attach(lambda: temperature_ramp(md.dyn.get_number_of_steps()), interval=temp_interval)

        md.run(steps)
        
        energy = calc_structure.get_potential_energy()

    else:
        raise ValueError("Invalid run_type. Choose from 'geo_opt' or 'NPT'.")

    return calc_structure, energy  # Return the updated structure

class Optimizer:
    """
    A class for sequential structure optimization using different calculation methods.
    """

    def __init__(self, structure: Union[Atoms, Structure]):
        """
        Initialize the Optimizer class.

        Args:
            structure (Union[Atoms, Structure]): Initial structure as an ASE Atoms object or Pymatgen Structure object.

        Raises:
            TypeError: If the provided structure is not ASE Atoms or Pymatgen Structure.
            ValueError: If the structure contains zero atoms.
        """
        # Convert Pymatgen structure to ASE Atoms if necessary
        if isinstance(structure, Atoms):
            self.structure = structure
        elif hasattr(structure, "to") and "pymatgen" in str(type(structure)):
            self.structure = AseAtomsAdaptor.get_atoms(structure)
        else:
            raise TypeError("The structure must be an ASE Atoms or Pymatgen Structure object.")

        # Raise an error if the structure has zero atoms
        if len(self.structure) == 0:
            raise ValueError("Error: The provided structure contains zero atoms. Please provide a valid structure.")

        self.optimizers = []  # List to store optimizer function configurations (as dictionaries)

    def set_structure(self, structure: Union[Atoms, Structure]):
        """
        Set a new structure for optimization.

        Args:
            structure (Union[Atoms, Structure]): New structure as an ASE Atoms object or Pymatgen Structure object.

        Raises:
            TypeError: If the provided structure is not ASE Atoms or Pymatgen Structure.
            ValueError: If the structure contains zero atoms.
        """
        if isinstance(structure, Atoms):
            self.structure = structure
        elif hasattr(structure, "to") and "pymatgen" in str(type(structure)):
            self.structure = AseAtomsAdaptor.get_atoms(structure)
        else:
            raise TypeError("The structure must be an ASE Atoms or Pymatgen Structure object.")

        if len(self.structure) == 0:
            raise ValueError("Error: The provided structure contains zero atoms. Please provide a valid structure.")

    def get_structure(self) -> Atoms:
        """
        Get the current structure.

        Returns:
            Atoms: The current ASE Atoms object.
        """
        return self.structure

    def set_optimizer(self, *optimizer_configs: dict):
        """
        Set the sequence of optimization functions with their respective parameters.

        Args:
            *optimizer_configs (dict): A list of dictionaries, each containing:
                - "run_function": The function to execute
                - Other keys as function parameters
        """
        for config in optimizer_configs:
            if not isinstance(config, dict) or "run_function" not in config:
                raise TypeError("Each optimizer configuration must be a dictionary containing 'run_function'.")

        self.optimizers = list(optimizer_configs)

    def get_optimizer(self):
        """
        Get the current optimizer sequence.

        Returns:
            List: The list of registered optimizer function configurations.
        """
        return self.optimizers

    def run(self):
        """
        Execute the optimizer sequence.

        Runs the registered optimization functions sequentially, updating the structure at each step.

        Returns:
            Atoms: The optimized ASE Atoms object.

        Raises:
            ValueError: If no optimizer sequence is set.
        """
        if not self.optimizers:
            raise ValueError("Optimizer sequence is not set.")

        for config in self.optimizers:
            optimizer_func = config.pop("run_function")  # Extract function
            if not callable(optimizer_func):
                raise TypeError(f"{optimizer_func} is not a valid function.")

            print(f"Running: {optimizer_func.__name__} with parameters: {config}",flush=True)

            # Run the optimizer function and update structure
            self.structure, _ = optimizer_func(self.structure, **config)

        return self.structure  # Return the final optimized structure
