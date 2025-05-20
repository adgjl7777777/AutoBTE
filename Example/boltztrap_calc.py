from AutoBTE.BoltzTrap.data_plot import bte_run
bte_run(
    output_dir="./test",
    vasp_dir="./vasp",
    fixed_temp=300,
    equivalence=20000,
    fixed_mu=0.0,
    plot_type=['dos','band_structure','electric_conductivity', 'seebeck', 'thermal_conductivity', 'powfactor', 'zT']
)