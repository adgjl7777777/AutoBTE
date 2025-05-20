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
        "run_function": chgnet_run,
        "run_type": "NPT",
        "cell_opt": True,
        "directory": f"./chgnet/",
        "Ti":1500,
        "Tf":300,
        "dt":1,
        "steps":750000,
        "ramp_type":"short"
    },
    {
        "run_function": vasp_run,
        "run_type": "geo_opt",
        "cell_opt": True,
        "directory": f"./vasp_opt1/",
        "system_name": "OPT_1",
        "k_point":(1,1,4),
        "E_cut":450,
        "cores":1,
        "vasp_kwargs": {
            "nsw": 100,
        }
    },
    {
        "run_function": vasp_run,
        "run_type": "geo_opt",
        "cell_opt": True,
        "directory": f"./vasp_opt2/",
        "system_name": "OPT_2",
        "k_point":(2,2,8),
        "E_cut":450,
        "cores":1
    }
)

# Run the optimization process
optimized_structure = opt.run()