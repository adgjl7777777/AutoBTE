# Import necessary functions
from ase.build import bulk
from AutoBTE.optimizer import Optimizer
from AutoBTE.optimizer import emt_run, vasp_run, chgnet_run
# Create an initial structure (e.g., Copper bulk)
atoms = bulk("Cu", "fcc", a=3.61)

# Initialize optimizer with structure and CPU core count
opt = Optimizer(atoms)

# Set optimizer sequence using dictionaries
opt.set_optimizer(
    {
        "run_function": emt_run,
        "run_type": "single",
        "directory": "./test/emt_1/"
    },
    {
        "run_function": vasp_run,
        "run_type": "geo_opt",
        "system_name": "Cu_Opt",
        "k_point": (6, 6, 6),
        "E_cut": 500,
        "cores": 4,
        "lreal": "Auto",
        "directory": "./test/vasp_1/"
    },
    {
        "run_function": chgnet_run,
        "run_type": "geo_opt",
        "directory": "./test/chgnet_1/"
    },
    {
        "run_function": vasp_run,
        "run_type": "NPT",
        "system_name": "Cu_MD",
        "k_point": (6, 6, 6),
        "E_cut": 500,
        "cores": 4,
        "Ti": 1500,
        "Tf": 298,
        "steps": 10000,
        "lreal": "Auto",
        "directory": "./test/vasp_2/"
    }
)

# Run the optimization process
optimized_structure = opt.run()