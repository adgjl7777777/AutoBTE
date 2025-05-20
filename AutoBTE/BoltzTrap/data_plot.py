import os
import sys
from ase.io import read,write
from AutoBTE.BoltzTrap.BTE import btp2

import numpy as np
import matplotlib.pyplot as plt

graph_axis_dict= {'electric_conductivity':'sigma', 'seebeck':'seebeck', 'thermal_conductivity':'kappa', 'powfactor':'powfactor', 'zT':'zT'}
def bte_run(output_dir,vasp_dir,fixed_temp=300,fixed_mu=0.0,plot_type=['dos','band_structure','electric_conductivity', 'seebeck', 'thermal_conductivity', 'powfactor', 'zT']):
    if not os.path.exists(f"{vasp_dir}/OUTCAR"):
        print(f"OUTCAR file not found in {vasp_dir}. Skipping this directory.")
        return
    # Read the structure from the VASP output file
    os.makedirs(output_dir, exist_ok=True)
    log_file = f"{output_dir}/boltztrap.log"

    for eachplot in plot_type:
        #calculate
        calc = btp2(vasp_dir)
        calc.interpolate(eq=20000)
        tmin, tmax, tsteps = 300, 1000, 20
        mu_range = [-5, 5]
        fermi_e = calc.get_fermi_energy()
        with open(log_file, 'a') as f:
            f.write(f"계산 시작\n")
            f.write(f"온도 범위: {tmin}K ~ {tmax}K, steps: {tsteps}\n")
            f.write(f"화학퍼텐셜 범위: {mu_range[0]} ~ {mu_range[1]} eV\n")
            f.write(f"총 포인트 수 : {len(calc.equivalences)}\n")
            f.write(f"Fermi energy: {fermi_e:.4f} eV\n\n")
        if eachplot not in ['dos','band_structure','electric_conductivity', 'seebeck', 'thermal_conductivity', 'powfactor', 'zT']:
            print(f"Invalid plot type: {eachplot}. Skipping this plot.")
            continue
        elif eachplot == 'dos':
            os.makedirs(f"{output_dir}/dos",exist_ok=True)
            calc.DOS.plot(plot="Smoothed", save=True, path=f"{output_dir}/dos/dos.png")
            epsilon_dos = calc.DOS.get_smoothed_epsilon()
            dos_values  = calc.DOS.get_smoothed_dos()
            np.save(f"{output_dir}/dos/epsilon.npy", epsilon_dos)
            np.save(f"{output_dir}/dos/dos.npy", dos_values)
            print(f"DOS calculation completed.",flush=True)
        elif eachplot == 'band_structure':
            os.makedirs(f"{output_dir}/bandstructure", exist_ok=True)
            try:
                calc.BandStructure.plot(save=True, path=f"{output_dir}/bandstructure/bandstructure.png")
                k_dist = calc.BandStructure.get_kpoints_distances()
                egrid  = calc.BandStructure.egrid
                np.save(f"{output_dir}/bandstructure/k_dist.npy", k_dist)
                np.save(f"{output_dir}/bandstructure/egrid.npy", egrid)
            except:
                print("band structure calculation failed.")
                continue
            print(f"band structure calculation completed.",flush=True)
        else:
            os.makedirs(f"{output_dir}/{eachplot}/T",exist_ok=True)
            os.makedirs(f"{output_dir}/{eachplot}/mu",exist_ok=True)
      
            for comp in ['x', 'y', 'z',None]:
                savecomp = comp if comp is not None else "all"
                # T 방향
                cx, cy = calc.FermiIntegrals.plot(
                    x="temperature", y=graph_axis_dict[eachplot], component=comp,
                    temp_range=(tmin, tmax), fixed_mu=fixed_mu,
                    save=True, path=f"{output_dir}/{eachplot}/T/{savecomp}.png"
                )
                np.save(f"{output_dir}/{eachplot}/T/xvalue_{savecomp}.npy",cx)
                np.save(f"{output_dir}/{eachplot}/T/yvalue_{savecomp}.npy",cy)
                # μ 방향
                cx, cy = calc.FermiIntegrals.plot(
                    x="chemical_potential", y=graph_axis_dict[eachplot], component=comp,
                    mu_range=mu_range, fixed_temp=fixed_temp,
                    save=True, path=f"{output_dir}/{eachplot}/mu/{savecomp}.png"
                )
                np.save(f"{output_dir}/{eachplot}/mu/xvalue_{savecomp}.npy",cx)
                np.save(f"{output_dir}/{eachplot}/mu/yvalue_{savecomp}.npy",cy)

            print(f"{eachplot} calculation complete.",flush=True)